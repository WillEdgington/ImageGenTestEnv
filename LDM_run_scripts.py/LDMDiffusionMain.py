import torch
import os
import sys

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data import prepareData
from utils.save import loadResultsMap, loadResultsMap, loadStates, saveModelAndResultsMap
from utils.visualize import plotDiffusionSamples, plotDiffusionTtraversalSamples, plotForwardDiffusion
from utils.losses import plotDiffusionLoss
from models.LDM import LDMVAE
from models.diffusion import UNet, LinearNoiseScheduler, CosineNoiseScheduler
from train.trainDiffusion import train

# General params
device = "cuda" if torch.cuda.is_available() else "cpu"
MANUALSEED = 42
DATA = "STANFORDCARS" # "CIFAR10" "STANFORDCARS" "CELEBA"
IMGSIZE = 64
IMGCHANNELS = 3
AUGMENT = 0.2 # preferably between 0 and 1

BATCHSIZE = 256
EPOCHS = 200
SAVEPOINT = 10
lrsf = 5
LR = (1 * pow(10, -lrsf) * (BATCHSIZE / 64))
WEIGHTDECAY = (1e-4 * (BATCHSIZE / 64))

datatag = DATA + str(IMGSIZE) if DATA != "CIFAR10" else DATA
lrtag = f"LR{lrsf}" if lrsf != 4 else ""
augtag = f"AUG{int(AUGMENT * 10)}" if AUGMENT != 0 else ""

trainParams = {"seed": MANUALSEED,
               "data": DATA,
               "image_size": IMGSIZE,
               "image_channels": IMGCHANNELS,
               "batch_size": BATCHSIZE,
               "learning_rate": LR,
               "weight_decay": WEIGHTDECAY}

# Autoencoder params
VAEBATCHSIZE = 64
VAEBASECHANNELS = 64
VAELATENTCHANNELS = 8
VAENUMDOWN = 3
VAERESBLOCKS = (2, 2)
VAENUMRESCONVS = (2, 2)
VAEISSTOCHASTIC = True
VAEEPOCHS = 100
stochtag = "STOCH" if VAEISSTOCHASTIC else ""
VAENAME = f"LDMVAE{datatag}BC{VAEBASECHANNELS}LC{VAELATENTCHANNELS}ND{VAENUMDOWN}RBE{VAERESBLOCKS[0]}RBD{VAERESBLOCKS[1]}NRCE{VAENUMRESCONVS[0]}NRCE{VAENUMRESCONVS[1]}BS{VAEBATCHSIZE}{stochtag}_{VAEEPOCHS}_EPOCHS_MODEL.pth"

ldmVAEParams = {"baseChannels": VAEBASECHANNELS,
                "vaeBatchSize": VAEBATCHSIZE,
                "latentChannels": VAELATENTCHANNELS,
                "numDown": VAENUMDOWN,
                "resBlocks": VAERESBLOCKS,
                "numResConvs": VAENUMRESCONVS,
                "isStochastic": VAEISSTOCHASTIC,
                "epochs": VAEEPOCHS,
                "name": VAENAME}

# Diffusion params
DIFBASECHANNELS = 128
DIFTIMEEMBDIM = None
DIFDEPTH = 2
DIFRESBLOCKS = (4, 8, 4)
DIFENCHEADS = 8
DIFDECHEADS = 16
DIFBOTHEADS = 32
DIFENCHEADDROP = 0.1
DIFDECHEADDROP = 0.1
DIFBOTHEADDROP = 0.1
TIMESTEPS = 1000
NSCHEDULE = "Cosine"
noiseScheduler = CosineNoiseScheduler(timesteps=TIMESTEPS) if NSCHEDULE == "Cosine" else LinearNoiseScheduler(timesteps=TIMESTEPS)

difinfotag = f"LDMDIFFUSION{datatag}{NSCHEDULE}T{TIMESTEPS}BS{BATCHSIZE}D{DIFDEPTH}BC{DIFBASECHANNELS}ERB{DIFRESBLOCKS[0]}EAH{DIFENCHEADS}EAD{int(DIFENCHEADDROP*10)}BRB{DIFRESBLOCKS[1]}BAH{DIFBOTHEADS}BAD{int(DIFBOTHEADDROP*10)}DRB{DIFRESBLOCKS[2]}DAH{DIFDECHEADS}EAD{int(DIFDECHEADDROP*10)}{lrtag}{augtag}"
DIFRESULTSNAME = f"{difinfotag}_RESULTS.pth"
DIFMODELNAME = f"{difinfotag}"

if __name__=="__main__":
    trainDataloader = prepareData(data=DATA, train=True, batchSize=BATCHSIZE, numWorkers=0, seed=MANUALSEED, imgSize=IMGSIZE, augment=AUGMENT)
    testDataloader = prepareData(data=DATA, train=False, batchSize=BATCHSIZE, numWorkers=0, imgSize=IMGSIZE)

    results = loadResultsMap(resultsName=DIFRESULTSNAME)
    epochscomplete = len(results["train_loss"]) if results is not None else 0

    torch.manual_seed(MANUALSEED)
    ldmvae = LDMVAE(imgChannels=IMGCHANNELS,
                    baseChannels=VAEBASECHANNELS,
                    latentChannels=VAELATENTCHANNELS,
                    numDown=VAENUMDOWN,
                    resBlocks=VAERESBLOCKS,
                    numResConvs=VAENUMRESCONVS,
                    stochastic=VAEISSTOCHASTIC)
    ldmvae.to(device)
    ldmvaestates = loadStates(stateName=VAENAME, model=ldmvae)
    if ldmvae is None:
        raise ValueError(f"{VAENAME} is not a valid model name")
    ldmvae.to(device)

    for param in ldmvae.parameters():
        param.requires_grad = False
    ldmvae.eval()

    torch.manual_seed(MANUALSEED)
    unet = UNet(imgInChannels=VAELATENTCHANNELS,
                imgOutChannnels=VAELATENTCHANNELS,
                timeEmbDim=DIFTIMEEMBDIM,
                depth=DIFDEPTH,
                resBlocks=DIFRESBLOCKS,
                baseChannels=DIFBASECHANNELS,
                numEncHeads=DIFENCHEADS,
                numDecHeads=DIFDECHEADS,
                numBotHeads=DIFBOTHEADS,
                encHeadsDropout=DIFENCHEADDROP,
                decHeadsDropout=DIFDECHEADDROP,
                botHeadsDropout=DIFBOTHEADDROP)
    unet.to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=WEIGHTDECAY)
    gradClipping = 1.0
    trainParams["grad_clipping"] = gradClipping
    
    states = {}
    states["ldmvae_params"] = ldmVAEParams
    states["train_params"] = trainParams
    if epochscomplete > 0:
        states = loadStates(stateName=DIFMODELNAME+f"_{epochscomplete}_EPOCHS_MODEL.pth",
                            model=unet, optimizer=optimizer)
    unet.to(device)
    gradClipping = states["train_params"]["grad_clipping"]

    lossFn = torch.nn.MSELoss(reduction="mean")

    summary(model=unet,
            input_size=[(BATCHSIZE, VAELATENTCHANNELS, IMGSIZE >> VAENUMDOWN, IMGSIZE >> VAENUMDOWN),(1,)],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    plotForwardDiffusion(dataloader=testDataloader,
                         noiseScheduler=noiseScheduler,
                         autoencoder=ldmvae,
                         numSamples=4,
                         step=TIMESTEPS//10,
                         title="",
                         classLabel=False,
                         seed=MANUALSEED,
                         device=device)

    while epochscomplete < EPOCHS:
        results = train(model=unet,
                        trainDataloader=trainDataloader,
                        testDataloader=testDataloader,
                        optimizer=optimizer,
                        lossFn=lossFn,
                        noiseScheduler=noiseScheduler,
                        epochs=SAVEPOINT,
                        autoencoder=ldmvae,
                        results=results,
                        numGeneratedSamples=5,
                        imgShape=(VAELATENTCHANNELS, IMGSIZE >> VAENUMDOWN, IMGSIZE >> VAENUMDOWN),
                        sampleEta=1.0,
                        seed=MANUALSEED,
                        device=device,
                        enableAmp=True,
                        gradClipping=gradClipping
                        )
        
        epochscomplete = len(results["train_loss"])
        states["train_params"]["epochs"] = epochscomplete
        states["model"] = unet.state_dict()
        states["optimizer"] = optimizer.state_dict()

        saveModelAndResultsMap(model=states, results=results, modelName=DIFMODELNAME+f"_{epochscomplete}_EPOCHS_MODEL.pth",
                               resultsName=DIFRESULTSNAME)
        
    plotDiffusionLoss(results=results, log=True,
                      step=1)
    plotDiffusionSamples(results=results, 
                         step=EPOCHS//10)
    plotDiffusionTtraversalSamples(model=unet,
                                   noiseScheduler=noiseScheduler,
                                   autoencoder=ldmvae,
                                   numSamples=5,
                                   imgShape=(VAELATENTCHANNELS, IMGSIZE >> VAENUMDOWN, IMGSIZE >> VAENUMDOWN),
                                   step=TIMESTEPS//10,
                                   skip=5,
                                   eta=1,
                                   title="",
                                   seed=MANUALSEED,
                                   device=device)
    