import torch
import sys

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data import prepareData
from models.vae import VAE, vaeLoss, AdaptiveMomentBetaScheduler
from train.trainVae import train
from utils.losses import plotVAELoss, plotVAELossAndBeta
from utils.visualize import plotVAEDecoderSamples
from utils.save import saveModelAndResultsMap

device = "cuda" if torch.cuda.is_available() else "cpu"

MANUALSEED = 42
BATCHSIZE = 32
EPOCHS = 100
BETA = 1.0
LR = 1e-4
EXTRACONVS = 15
LATENTDIMS = 50
SCHEDULER = True

ADAPTIVESTR = "_ADAPTIVE_" if SCHEDULER else ""
MODELNAME = f"VAE_CIFAR10_{int(BETA)}_BETA{ADAPTIVESTR}_{LATENTDIMS}_ZDIMS_{EXTRACONVS}_CONVS"
RESULTSNAME = f"VAE_CIFAR10_{int(BETA)}_BETA{ADAPTIVESTR}_{LATENTDIMS}_ZDIMS_{EXTRACONVS}_CONVS_RESULTS.pth"

if __name__=="__main__":
    # create CIFAR-10 dataloaders
    trainDataloader = prepareData(batchSize=BATCHSIZE, seed=MANUALSEED)
    testDataloader = prepareData(train=False, batchSize=BATCHSIZE)

    torch.manual_seed(MANUALSEED)
    vae = VAE(latentDim=LATENTDIMS, upAddConv=EXTRACONVS, downAddConv=EXTRACONVS, downConvNorm=True, upConvNorm=True)
    vae.to(device)


    summary(model=vae,
            input_size=(1, 3, 32, 32),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    betaScheduler = AdaptiveMomentBetaScheduler()

    optimizer = torch.optim.Adam(vae.parameters(), lr=LR)
    completedEpochs = 0
    VAEresults = None

    while completedEpochs != EPOCHS:
        torch.manual_seed(MANUALSEED)
        VAEresults = train(model=vae,
                        trainDataloader=trainDataloader,
                        testDataloader=testDataloader,
                        optimizer=optimizer,
                        epochs=EPOCHS,
                        beta=betaScheduler.beta,
                        device=device,
                        latentDim=LATENTDIMS,
                        betaScheduler=betaScheduler,
                        results=VAEresults,
                        decSamplesPerEpoch=5)
        
        # Save the results and model
        saveModelAndResultsMap(model=vae, 
                                results=VAEresults, 
                                modelName=MODELNAME + f"_{completedEpochs}_EPOCHS_MODEL.pth",
                                resultsName=RESULTSNAME)

    # Plot loss curves for VAE
    # plotVAELoss(results=VAEresults)
    plotVAELoss(results=VAEresults)

    # Plot the generated images from training loop
    plotVAEDecoderSamples(results=VAEresults, step=10)
