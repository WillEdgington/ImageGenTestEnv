import torch
import sys
import os

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data import prepareData
from utils.save import saveModelAndResultsMap, loadResultsMap, loadModel
from utils.visualize import plotDiffusionSamples, plotDiffusionTtraversalSamples, plotForwardDiffusion
from utils.losses import plotDiffusionLoss
from models.diffusion import UNet, LinearNoiseScheduler, CosineNoiseScheduler
from train.trainDiffusion import train

device = "cuda" if torch.cuda.is_available() else "cpu"
MANUALSEED = 42

BASECHANNELS = 64
IMGCHANNELS = 3
TIMEEMBDIM = None
DEPTH = 3
ENCHEADS = 2
DECHEADS = 2
BOTHEADS = 2
ENCHEADDROP = 0.1
DECHEADDROP = 0.1
BOTHEADDROP = 0.1
TIMESTEPS = 1000

LR = 1e-4
BATCHSIZE = 64
EPOCHS = 300
SAVEPOINT = 10

RESULTSNAME = f"DIFFUSION_CIFAR10_RESULTS.pth"
MODELNAME = f"DIFFUSION_CIFAR10"

if __name__=="__main__":
    trainDataloader = prepareData(batchSize=BATCHSIZE, seed=MANUALSEED)
    testDataloader = prepareData(train=False, batchSize=BATCHSIZE)

    results = loadResultsMap(resultsName=RESULTSNAME)
    epochscomplete = len(results["train_loss"]) if results is not None else 0

    torch.manual_seed(MANUALSEED)
    unet = UNet(imgInChannels=IMGCHANNELS,
                imgOutChannnels=IMGCHANNELS,
                timeEmbDim=TIMEEMBDIM,
                depth=DEPTH,
                baseChannels=BASECHANNELS,
                numEncHeads=ENCHEADS,
                numDecHeads=DECHEADS,
                numBotHeads=BOTHEADS,
                encHeadsDropout=ENCHEADDROP,
                decHeadsDropout=DECHEADDROP,
                botHeadsDropout=BOTHEADDROP)

    if epochscomplete > 0:
        unet = loadModel(model=unet, modelName=MODELNAME+f"_{epochscomplete}_EPOCHS_MODEL.pth", device=device)

    summary(model=unet,
            input_size=[(1, 3, 32, 32),(1,)],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    lossFn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=unet.parameters(), lr=LR)
    noiseScheduler = CosineNoiseScheduler(timesteps=TIMESTEPS)

    plotForwardDiffusion(dataloader=testDataloader,
                         noiseScheduler=noiseScheduler,
                         numSamples=3,
                         step=100,
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
                        imgShape=(3,32,32),
                        sampleEta=1.0,
                        seed=MANUALSEED,
                        device=device)
        
        epochscomplete = len(results["train_loss"])

        saveModelAndResultsMap(model=unet, results=results, modelName=MODELNAME+f"_{epochscomplete}_EPOCHS_MODEL.pth", resultsName=RESULTSNAME)


    plotDiffusionLoss(results=results)
    plotDiffusionSamples(results=results, step=10)
    plotDiffusionTtraversalSamples(model=unet,
                                   noiseScheduler=noiseScheduler,
                                   numSamples=3,
                                   imgShape=(3, 32, 32),
                                   step=100,
                                   skip=1,
                                   eta=1.0,
                                   title="",
                                   seed=MANUALSEED,
                                   device=device)
    plotDiffusionTtraversalSamples(model=unet,
                                   noiseScheduler=noiseScheduler,
                                   numSamples=3,
                                   imgShape=(3, 32, 32),
                                   step=100,
                                   skip=20,
                                   eta=1.0,
                                   title="",
                                   seed=MANUALSEED,
                                   device=device)