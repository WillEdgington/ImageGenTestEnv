import torch
import sys
import os

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data import prepareData
from utils.save import saveModelAndResultsMap, loadResultsMap, loadModel, loadStates, deleteModel
from models.LDM import LDMVAE
from models.vae import AdaptiveMomentBetaScheduler
from train.trainVae import train
from utils.visualize import plotVAEDecoderSamples, visualiseVAELatentTraversal
from utils.losses import plotVAELossAndBeta, plotVAELossGradients

device = "cuda" if torch.cuda.is_available() else "cpu"
MANUALSEED = 42
DATA = "STANFORDCARS" # "CIFAR10" "STANFORDCARS" "CELEBA"
IMGSIZE = 128
IMGCHANNELS = 3
AUGMENT = 0.2

BASECHANNELS = 64
LATENTCHANNELS = 12
NUMDOWN = 4
RESBLOCKS = (2, 2)
NUMRESCONVS = (2, 2)
ISSTOCHASTIC = True

BATCHSIZE = 32
EPOCHS = 100
SAVEPOINT = 10
lrsf = 4
LR = (1 * pow(10, -lrsf) * (BATCHSIZE / 64))
WEIGHTDECAY = (1e-4 * (BATCHSIZE / 64))

lrtag = f"LR{lrsf}" if lrsf != 4 else ""
augtag = f"AUG{int(AUGMENT * 10)}" if AUGMENT != 0 else ""
datatag = DATA + str(IMGSIZE) if DATA != "CIFAR10" else DATA
stochtag = "STOCH" if ISSTOCHASTIC else ""
infostr = f"LDMVAE{datatag}BC{BASECHANNELS}LC{LATENTCHANNELS}ND{NUMDOWN}RBE{RESBLOCKS[0]}RBD{RESBLOCKS[1]}NRCE{NUMRESCONVS[0]}NRCE{NUMRESCONVS[1]}BS{BATCHSIZE}{lrtag}{augtag}{stochtag}"
RESULTSNAME = f"{infostr}_RESULTS.pth"
MODELNAME = f"{infostr}"

if __name__=="__main__":
    trainDataloader = prepareData(data=DATA, batchSize=BATCHSIZE, numWorkers=0, seed=MANUALSEED, imgSize=IMGSIZE, augment=AUGMENT)
    testDataloader = prepareData(data=DATA, train=False, batchSize=BATCHSIZE, numWorkers=0, imgSize=IMGSIZE)

    results = loadResultsMap(resultsName=RESULTSNAME)
    epochscomplete = len(results["train_loss"]) if results is not None else 0
    
    torch.manual_seed(MANUALSEED)
    ldmvae = LDMVAE(imgChannels=IMGCHANNELS,
                    baseChannels=BASECHANNELS,
                    latentChannels=LATENTCHANNELS,
                    numDown=NUMDOWN,
                    resBlocks=RESBLOCKS,
                    numResConvs=NUMRESCONVS,
                    stochastic=ISSTOCHASTIC).to(device)
    
    betaScheduler = AdaptiveMomentBetaScheduler(betaInit=1e-6,
                                                gamma=0.1,
                                                eta=1,
                                                betaMin=1e-8,
                                                betaMax=1,
                                                warmup=5)
    optimizer = torch.optim.AdamW(ldmvae.parameters(), lr=LR, weight_decay=WEIGHTDECAY)

    states = {}
    if epochscomplete > 0:
        states = loadStates(stateName=MODELNAME+f"_{epochscomplete}_EPOCHS_MODEL.pth", 
                            model=ldmvae, optimizer=optimizer, betaScheduler=betaScheduler)
    ldmvae.to(device)
    
    summary(model=ldmvae,
            input_size=[(BATCHSIZE, IMGCHANNELS, IMGSIZE, IMGSIZE)],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    while epochscomplete < EPOCHS:
        results = train(model=ldmvae,
                        trainDataloader=trainDataloader,
                        testDataloader=testDataloader,
                        optimizer=optimizer,
                        epochs=SAVEPOINT,
                        beta=betaScheduler.beta,
                        enableAmp=True,
                        device=device,
                        latentDim=IMGSIZE >> NUMDOWN,
                        decSamplesPerEpoch=5,
                        countActiveDims=True,
                        betaScheduler=betaScheduler,
                        results=results)
        
        states["model"] = ldmvae.state_dict()
        states["optimizer"] = optimizer.state_dict()
        states["betaScheduler"] = betaScheduler.state_dict()
        epochscomplete += SAVEPOINT

        saveModelAndResultsMap(model=states, results=results, modelName=MODELNAME+f"_{epochscomplete}_EPOCHS_MODEL.pth",
                               resultsName=RESULTSNAME)
        
        if epochscomplete > 100:
            deleteModel(modelName=MODELNAME+f"_{epochscomplete-100}_EPOCHS_MODEL.pth")

    plotVAEDecoderSamples(results=results, step=EPOCHS//10)
    plotVAELossAndBeta(results=results)
    plotVAELossGradients(results=results, alpha=0.3)
    visualiseVAELatentTraversal(vae=ldmvae,
                                testDataloader=testDataloader,
                                numSamples=5,
                                latentIdx=(5,0,0),
                                steps=11)
