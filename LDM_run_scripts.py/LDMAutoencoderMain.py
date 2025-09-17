import torch
import sys
import os

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data import prepareData
from utils.save import saveModelAndResultsMap, loadResultsMap, loadModel
from models.LDM import LDMVAE
from models.vae import vaeLoss, AdaptiveMomentBetaScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
MANUALSEED = 42
DATA = "CIFAR10" # "CIFAR10" "STANFORDCARS" "CELEBA"
IMGSIZE = 32
IMGCHANNELS = 3

BASECHANNELS = 64
LATENTCHANNELS = 4
NUMDOWN = 3
RESBLOCKS = (2, 2)
NUMRESCONVS = (2, 2)
ISSTOCHASTIC = True

BATCHSIZE = 64
EPOCHS = 100
SAVEPOINT = 10
LR = (1e-4 * (BATCHSIZE / 64))

if __name__=="__main__":
    trainDataloader = prepareData(data=DATA, batchSize=BATCHSIZE, numWorkers=0, seed=MANUALSEED, imgSize=IMGSIZE)
    testDataloader = prepareData(data=DATA, train=False, batchSize=BATCHSIZE, numWorkers=0, imgSize=IMGSIZE)

    torch.manual_seed(MANUALSEED)
    ldmvae = LDMVAE(imgChannels=IMGCHANNELS,
                    baseChannels=BASECHANNELS,
                    latentChannels=LATENTCHANNELS,
                    numDown=NUMDOWN,
                    resBlocks=RESBLOCKS,
                    numResConvs=NUMRESCONVS,
                    stochastic=ISSTOCHASTIC)
    ldmvae.to(device)
    
    summary(model=ldmvae,
            input_size=[(BATCHSIZE, IMGCHANNELS, IMGSIZE, IMGSIZE)],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    betaScheduler = AdaptiveMomentBetaScheduler(betaInit=1e-8,
                                                gamma=0.1,
                                                eta=1,
                                                betaMin=1e-8,
                                                betaMax=1,
                                                warmup=5)
    optimizer = torch.optim.Adam(ldmvae.parameters(), lr=LR)
    results = None