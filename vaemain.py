import torch

from torchinfo import summary

from utils.data import prepareData
from models.vae import VAE, vaeLoss
from train.trainVae import train
from utils.losses import plotVAELoss
from utils.visualize import plotVAEDecoderSamples
from utils.save import saveModelAndResultsMap

device = "cuda" if torch.cuda.is_available() else "cpu"

MANUALSEED = 42
BATCHSIZE = 32

if __name__=="__main__":
    # create CIFAR-10 dataloaders
    trainDataloader = prepareData(batchSize=BATCHSIZE, seed=MANUALSEED)
    testDataloader = prepareData(train=False, batchSize=BATCHSIZE)

    LATENTDIM = 100
    torch.manual_seed(MANUALSEED)
    # Create instance of Variational Auto Encoder (VAE) class
    vae = VAE(latentDim=LATENTDIM)
    vae.to(device)

    # # Get a summary of the VAE (uncomment to see)
    # summary(model=vae,
    #         input_size=(1, 3, 32, 32),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])

    # set learning rate
    LR = 2e-4

    # Define optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=LR)

    # Train VAE model
    EPOCHS = 150

    torch.manual_seed(MANUALSEED)
    VAEresults = train(model=vae,
                       trainDataloader=trainDataloader,
                       testDataloader=testDataloader,
                       optimizer=optimizer,
                       epochs=EPOCHS,
                       device=device,
                       latentDim=LATENTDIM,
                       decSamplesPerEpoch=5)
    
    # Save the results and model
    saveModelAndResultsMap(model=vae, 
                           results=VAEresults, 
                           modelName=f"VAE_CIFAR10_{EPOCHS}_EPOCHS_MODEL.pth",
                           resultsName=f"VAE_CIFAR10_{EPOCHS}_EPOCHS_RESULTS.pth")

    # Plot loss curves for VAE
    plotVAELoss(results=VAEresults)

    # Plot the generated images from training loop
    plotVAEDecoderSamples(results=VAEresults, step=10)
