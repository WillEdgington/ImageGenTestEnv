import torch
import sys
import os

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data import prepareData
from utils.save import saveModelAndResultsMap, loadResultsMap, loadModel, loadStates
from utils.visualize import plotDiffusionSamples, plotDiffusionTtraversalSamples, plotForwardDiffusion, plotDiffusionSamplingFromNoisedData
from utils.losses import plotDiffusionLoss
from models.diffusion import UNet, LinearNoiseScheduler, CosineNoiseScheduler
from train.trainDiffusion import train

device = "cuda" if torch.cuda.is_available() else "cpu"
MANUALSEED = 42
DATA = "CIFAR10" # "CIFAR10" "STANFORDCARS" "CELEBA"
IMGSIZE = 32

BASECHANNELS = 64
IMGCHANNELS = 3
TIMEEMBDIM = None
DEPTH = 3
RESBLOCKS = (2, 2, 2)
ENCHEADS = 2
DECHEADS = 2
BOTHEADS = 2
ENCHEADDROP = 0.1
DECHEADDROP = 0.1
BOTHEADDROP = 0.1
TIMESTEPS = 1000
NSCHEDULE = "Cosine"
noiseScheduler = CosineNoiseScheduler(timesteps=TIMESTEPS) if NSCHEDULE == "Cosine" else LinearNoiseScheduler(timesteps=TIMESTEPS)

BATCHSIZE = 64
EPOCHS = 100
SAVEPOINT = 10
lrsf = 4
LR = (1 * pow(10, -lrsf) * (BATCHSIZE / 64))
WEIGHTDECAY = (1e-4 * (BATCHSIZE / 64))
AUGMENT = 0
gradClipping = 1.0

trainParams = {"seed": MANUALSEED,
               "data": DATA,
               "image_size": IMGSIZE,
               "image_channels": IMGCHANNELS,
               "batch_size": BATCHSIZE,
               "learning_rate": LR,
               "weight_decay": WEIGHTDECAY,
               "grad_clipping": gradClipping}

lrtag = f"LR{lrsf}" if lrsf != 4 else ""
augtag = f"AUG{int(AUGMENT * 10)}" if AUGMENT != 0 else ""
datatag = DATA + str(IMGSIZE) if DATA != "CIFAR10" else ""
infotag = f"DIFFUSION{datatag}{NSCHEDULE}T{TIMESTEPS}BS{BATCHSIZE}D{DEPTH}BC{BASECHANNELS}ERB{RESBLOCKS[0]}EAH{ENCHEADS}EAD{int(ENCHEADDROP*10)}BRB{RESBLOCKS[1]}BAH{BOTHEADS}BAD{int(BOTHEADDROP*10)}DRB{RESBLOCKS[2]}DAH{DECHEADS}EAD{int(DECHEADDROP*10)}{lrtag}{augtag}"
RESULTSNAME = f"{infotag}_RESULTS.pth"
MODELNAME = f"{infotag}"

if __name__=="__main__":
    trainDataloader = prepareData(data=DATA, batchSize=BATCHSIZE, numWorkers=0, seed=MANUALSEED, imgSize=IMGSIZE, augment=AUGMENT)
    testDataloader = prepareData(data=DATA, train=False, batchSize=BATCHSIZE, numWorkers=0, imgSize=IMGSIZE)

    results = loadResultsMap(resultsName=RESULTSNAME)
    epochscomplete = min(len(results["train_loss"]), EPOCHS) if results is not None else 0

    torch.manual_seed(MANUALSEED)
    unet = UNet(imgInChannels=IMGCHANNELS,
                imgOutChannnels=IMGCHANNELS,
                timeEmbDim=TIMEEMBDIM,
                depth=DEPTH,
                resBlocks=RESBLOCKS,
                baseChannels=BASECHANNELS,
                numEncHeads=ENCHEADS,
                numDecHeads=DECHEADS,
                numBotHeads=BOTHEADS,
                encHeadsDropout=ENCHEADDROP,
                decHeadsDropout=DECHEADDROP,
                botHeadsDropout=BOTHEADDROP)
    unet.to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=WEIGHTDECAY)

    states = {}
    states["train_params"] = trainParams
    if epochscomplete > 0:
        states = loadStates(stateName=MODELNAME+f"_{epochscomplete}_EPOCHS_MODEL.pth",
                            model=unet, optimizer=optimizer)
    unet.to(device)
    gradClipping = states["train_params"]["grad_clipping"]

    lossFn = torch.nn.MSELoss(reduction="mean")

    summary(model=unet,
            input_size=[(BATCHSIZE, 3, IMGSIZE, IMGSIZE),(1,)],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    plotForwardDiffusion(dataloader=testDataloader,
                         noiseScheduler=noiseScheduler,
                         numSamples=4,
                         step=TIMESTEPS//10,
                         title="",
                         seed=MANUALSEED)

    while epochscomplete < EPOCHS:
        results = train(model=unet,
                        trainDataloader=trainDataloader,
                        testDataloader=testDataloader,
                        optimizer=optimizer,
                        lossFn=lossFn,
                        noiseScheduler=noiseScheduler,
                        epochs=SAVEPOINT,
                        results=results,
                        numGeneratedSamples=5,
                        imgShape=(3,IMGSIZE,IMGSIZE),
                        sampleEta=1.0,
                        seed=MANUALSEED,
                        device=device,
                        enableAmp=True,
                        gradClipping=gradClipping)
        
        epochscomplete = len(results["train_loss"])
        states["train_params"]["epochs"] = epochscomplete
        states["model"] = unet.state_dict()
        states["optimizer"] = optimizer.state_dict()

        saveModelAndResultsMap(model=states, results=results, modelName=MODELNAME+f"_{epochscomplete}_EPOCHS_MODEL.pth",
                               resultsName=RESULTSNAME)

    plotDiffusionLoss(results=results, log=True)
    plotDiffusionSamples(results=results, step=EPOCHS//10)
    plotDiffusionTtraversalSamples(model=unet,
                                   noiseScheduler=noiseScheduler,
                                   numSamples=4,
                                   imgShape=(3, IMGSIZE, IMGSIZE),
                                   step=TIMESTEPS//10,
                                   skip=20,
                                   eta=1,
                                   title="",
                                   seed=MANUALSEED,
                                   device=device)
    plotDiffusionSamplingFromNoisedData(model=unet,
                                        dataloader=testDataloader,
                                        noiseScheduler=noiseScheduler,
                                        numSamples=3,
                                        step=TIMESTEPS//10,
                                        skip=20,
                                        eta=1,
                                        title="",
                                        classLabel=False,
                                        seed=MANUALSEED,
                                        device=device)