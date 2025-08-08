import torch
import sys

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.losses import plotVAELossAndBeta, plotVAELoss
from utils.visualize import plotVAEDecoderSamples, visualiseVAELatentTraversal, plotVAEDecoderSamples
from utils.save import loadModel, loadResultsMap
from utils.data import prepareData
from models.vae import VAE

MANUALSEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 100
BETA = 1.0
EXTRACONVS = 15
NORMCONVS = False
LATENTDIMS = 50
SCHEDULER = True
GAMMA=5e-2

NORMCONVSSTR = "_NORM" if NORMCONVS else ""
ADAPTIVESTR = "_ADAPTIVE_" + (f"{str(GAMMA)[2:]}_GAMMA" if GAMMA else "") if SCHEDULER else ""
MODELNAME = f"VAE_CIFAR10_{int(BETA)}_BETA{ADAPTIVESTR}_{LATENTDIMS}_ZDIMS_{EXTRACONVS}_CONVS{NORMCONVSSTR}_{EPOCHS}_EPOCHS_MODEL.pth"
RESULTSNAME = f"VAE_CIFAR10_{int(BETA)}_BETA{ADAPTIVESTR}_{LATENTDIMS}_ZDIMS_{EXTRACONVS}_CONVS{NORMCONVSSTR}_RESULTS.pth"

TITLE = f"VAE (Beta (initial): {BETA}, Latent dimensions: {LATENTDIMS}, Extra Convs: {EXTRACONVS}{NORMCONVSSTR}, AdaBeta: {SCHEDULER})"

if __name__=="__main__":
    testDataloader = prepareData(train=False)
    vae = loadModel(model=VAE(latentDim=LATENTDIMS, upAddConv=EXTRACONVS, downAddConv=EXTRACONVS, upConvNorm=NORMCONVS, downConvNorm=NORMCONVS), modelName=MODELNAME)
    vae.to(device)

    results = loadResultsMap(resultsName=RESULTSNAME)

    # Get summary of model
    summary(model=vae,
            input_size=(1, 3, 32, 32),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    plotVAELossAndBeta(results=results, title=TITLE)

    plotVAEDecoderSamples(results=results, step=10, title=TITLE)
    
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