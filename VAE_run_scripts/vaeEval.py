import torch
import sys

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.losses import plotVAELossAndBeta, plotVAELoss
from utils.visualize import plotVAEDecoderSamples, visualiseVAELatentTraversal
from utils.save import loadModel, loadResultsMap
from utils.data import prepareData
from models.vae import VAE

MANUALSEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

BETA = 1.0
SCHEDULER = True
LATENTDIMS = 50
EXTRACONVS = 5
EPOCHS = 100

ADAPTIVESTR = "_ADAPTIVE_" if SCHEDULER else ""
MODELNAME = f"VAE_CIFAR10_{int(BETA)}_BETA{ADAPTIVESTR}_{LATENTDIMS}_ZDIMS_{EXTRACONVS}_CONVS_{EPOCHS}_EPOCHS_MODEL.pth"
RESULTSNAME = f"VAE_CIFAR10_{int(BETA)}_BETA{ADAPTIVESTR}_{LATENTDIMS}_ZDIMS_{EXTRACONVS}_CONVS_RESULTS.pth"

TITLE = f"VAE (Beta (initial): {BETA}, Latent dimensions: {LATENTDIMS}, Extra Convs: {EXTRACONVS}, AdaBeta: {SCHEDULER})"

if __name__=="__main__":
    testDataloader = prepareData(train=False)
    vae = loadModel(model=VAE(latentDim=LATENTDIMS, upAddConv=EXTRACONVS, downAddConv=EXTRACONVS), modelName=MODELNAME)
    vae.to(device)

    results = loadResultsMap(resultsName=RESULTSNAME)

    # Get summary of model
    summary(model=vae,
            input_size=(1, 3, 32, 32),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    plotVAELossAndBeta(results=results, title=TITLE)
    
    std = results["latent_dims_std_train"][EPOCHS - 1]
    for i in range(LATENTDIMS):
        visualiseVAELatentTraversal(vae=vae,
                                    testDataloader=testDataloader,
                                    numSamples=5,
                                    latentIdx=i,
                                    steps=11,
                                    title=TITLE,
                                    minZ=-int(std * 10),
                                    maxZ=int(std * 10),
                                    seed=MANUALSEED,
                                    device=device)