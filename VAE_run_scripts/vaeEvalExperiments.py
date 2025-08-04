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

if __name__=="__main__":
    betas = [1.0, 4.0]
    useScheduler = [False, True]
    latentDims = [50]
    batchSizes = [32]
    learningRates = [1e-4]
    extraConvsArr = [5]
    
    for beta in betas:
        for scheduler in useScheduler:
            for latentDim in latentDims:
                for extraConvs in extraConvsArr:
                    adaptiveStr = "_ADAPTIVE_" if scheduler else ""
                    modelName = f"VAE_CIFAR10_{int(beta)}_BETA{adaptiveStr}_{latentDim}_ZDIMS_{extraConvs}_CONVS_"
                    resultsName = f"VAE_CIFAR10_{int(beta)}_BETA{adaptiveStr}_{latentDim}_ZDIMS_{extraConvs}_CONVS_RESULTS.pth"

                    results = loadResultsMap(resultsName=resultsName)

                    if results is None:
                        continue
                    
                    epochs = len(results["train_loss"])
                    modelName += f"{epochs}_EPOCHS_MODEL.pth"
                    vae = loadModel(model=VAE(latentDim=latentDim, upAddConv=extraConvs, downAddConv=extraConvs), modelName=modelName)

                    # Get summary of model
                    summary(model=vae,
                        input_size=(1, 3, 32, 32),
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"])

                    title = f"VAE (Beta (initial): {beta}, Latent dimensions: {latentDim}, Extra Convs: {extraConvs}, AdaBeta: {scheduler})"
                    plotVAELossAndBeta(results=results, title=title)
                    plotVAEDecoderSamples(results=results, step=10, title=title)
                    