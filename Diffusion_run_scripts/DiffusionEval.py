import torch
import sys
import os

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data import prepareData
from utils.save import loadResultsMap, loadModel
from utils.visualize import plotDiffusionSamples, plotDiffusionTtraversalSamples, plotForwardDiffusion
from utils.losses import plotDiffusionLoss
from models.diffusion import UNet, LinearNoiseScheduler, CosineNoiseScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
MANUALSEED = 42
DATA = "STANFORDCARS" # "CIFAR10" "STANFORDCARS" "CELEBA"
IMGSIZE = 64

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
NSCHEDULE = "Cosine" # "Cosine" "Linear"
noiseScheduler = CosineNoiseScheduler(timesteps=TIMESTEPS) if NSCHEDULE == "Cosine" else LinearNoiseScheduler(timesteps=TIMESTEPS)

BATCHSIZE = 4
EPOCHS = None # if None then get latest model

datatag = DATA + str(IMGSIZE) if DATA != "CIFAR10" else ""
RESULTSNAME = f"DIFFUSION{datatag}{NSCHEDULE}T{TIMESTEPS}BS{BATCHSIZE}D{DEPTH}BC{BASECHANNELS}AH{ENCHEADS}AD{int(ENCHEADDROP*10)}_CIFAR10_RESULTS.pth"
MODELNAME = f"DIFFUSION{datatag}{NSCHEDULE}T{TIMESTEPS}BS{BATCHSIZE}D{DEPTH}BC{BASECHANNELS}AH{ENCHEADS}AD{int(ENCHEADDROP*10)}_CIFAR10"
TITLE = f"{DATA} ({IMGSIZE}x{IMGSIZE}) Diffusion Model (Schedule: {NSCHEDULE}, T: {TIMESTEPS}, ENC/DEC depth: {DEPTH}, base channels: {BASECHANNELS})"

if __name__=="__main__":
    trainDataloader = prepareData(data=DATA, batchSize=BATCHSIZE, numWorkers=0, seed=MANUALSEED, imgSize=IMGSIZE)
    testDataloader = prepareData(data=DATA, train=False, batchSize=BATCHSIZE, numWorkers=0, imgSize=IMGSIZE)

    results = loadResultsMap(resultsName=RESULTSNAME)
    epochs = len(results["train_loss"])
    if EPOCHS is not None:
        epochs = min(EPOCHS, epochs)

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
    
    unet = loadModel(model=unet, modelName=MODELNAME+f"_{epochs}_EPOCHS_MODEL.pth", device=device)

    summary(model=unet,
            input_size=[(BATCHSIZE, 3, IMGSIZE, IMGSIZE),(1,)],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    plotForwardDiffusion(dataloader=testDataloader,
                         noiseScheduler=noiseScheduler,
                         numSamples=4,
                         step=TIMESTEPS//10,
                         title=f"{DATA}, {NSCHEDULE} scheduler",
                         seed=MANUALSEED)
    
    plotDiffusionLoss(results=results, log=True, title=TITLE)
    plotDiffusionSamples(results=results, step=epochs//10, title=TITLE)
    plotDiffusionTtraversalSamples(model=unet,
                                   noiseScheduler=noiseScheduler,
                                   numSamples=4,
                                   imgShape=(3, IMGSIZE, IMGSIZE),
                                   step=TIMESTEPS//10,
                                   skip=10,
                                   eta=1,
                                   title=TITLE,
                                   seed=MANUALSEED,
                                   device=device)