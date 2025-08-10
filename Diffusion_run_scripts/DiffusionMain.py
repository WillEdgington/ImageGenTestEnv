import torch
import sys
import os

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data import prepareData
from utils.save import saveModelAndResultsMap
from models.diffusion import UNet, LinearNoiseScheduler
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
ENCHEADDROP = 0
DECHEADDROP = 0
BOTHEADDROP = 0
TIMESTEPS = 10

LR = 2e-4
BATCHSIZE = 32
EPOCHS = 10

if __name__=="__main__":
    trainDataloader = prepareData(batchSize=BATCHSIZE, seed=MANUALSEED)
    testDataloader = prepareData(train=False, batchSize=BATCHSIZE)

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

    summary(model=unet,
            input_size=[(1, 3, 32, 32),(1,)],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    lossFn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=unet.parameters(), lr=LR)
    noiseScheduler = LinearNoiseScheduler(timesteps=TIMESTEPS)

    results = train(model=unet,
                    trainDataloader=trainDataloader,
                    testDataloader=testDataloader,
                    optimizer=optimizer,
                    lossFn=lossFn,
                    noiseScheduler=noiseScheduler,
                    epochs=EPOCHS,
                    results=None,
                    device=device)