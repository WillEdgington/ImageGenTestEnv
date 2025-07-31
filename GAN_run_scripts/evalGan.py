import torch

from torchinfo import summary

from utils.save import loadModel, loadResultsMap
from models.gan import Generator, Discriminator
from utils.visualize import plotGANGeneratorSamples
from utils.losses import plotGANLoss, plotGANTrainLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
MANUALSEED = 42

if __name__=="__main__":
    generator = loadModel(model=Generator(), modelName="GENERATOR_GAN_CIFAR10_150_EPOCHS_MODEL.pth")
    discriminator = loadModel(model=Discriminator(), modelName="DISCRIMINATOR_GAN_CIFAR10_150_EPOCHS_MODEL.pth")
    results = loadResultsMap(resultsName="GAN_CIFAR10_150_EPOCHS_RESULTS.pth")

    LATENTDIM=100
    IMGDIM = 32
    IMGCHANNELS = 3
    print("\nGenerator Summary")
    summary(generator, 
            input_size=(1, LATENTDIM, 1, 1),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    print("\nDiscriminator Summary")
    summary(discriminator, 
            input_size=(1, IMGCHANNELS, IMGDIM, IMGDIM),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    plotGANGeneratorSamples(results, step=10)
    plotGANTrainLoss(results)
