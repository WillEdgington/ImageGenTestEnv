import torch
import matplotlib.pyplot as plt
import sys

from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.gan import Generator, Discriminator
from train.trainGan import train
from utils.losses import plotGANLoss
from utils.visualize import plotGANGeneratorSamples
from utils.save import saveModelAndResultsMap, saveGANandResultsMap
from utils.data import prepareData

device = "cuda" if torch.cuda.is_available() else "cpu"

# constants
MANUALSEED = 42
BATCHSIZE = 32

if __name__=="__main__":
    # Prepare data loaders
    trainDataloader = prepareData(batchSize=BATCHSIZE, seed=MANUALSEED)
    testDataloader = prepareData(train=False, batchSize=BATCHSIZE)

    LATENTDIM = 100
    torch.manual_seed(MANUALSEED)
    # Create instance of generator for GAN model
    generator = Generator(latentDim=LATENTDIM, imgChannels=3, featureMapSize=64)
    generator.to(device) # send to chosen device (GPU if possible)

    # # Get a summary of generator (uncomment to see)
    # summary(generator, 
    #         input_size=(1, 100, 1, 1),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])

    torch.manual_seed(MANUALSEED)
    # Create instance of discriminator for GAN model
    discriminator = Discriminator(imgChannels=3, featureMapSize=64)
    discriminator.to(device)

    # # Get a summary of discriminator (uncomment to see)
    # summary(discriminator, 
    #         input_size=(1, 3, 32, 32),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])

    # Create loss function (Binary cross entropy) and optimizer (Adam with lr = B1 = 0.5)
    GANloss = nn.BCELoss()

    # Using learning rate a Beta 1 values proposed in DCGAN paper (https://arxiv.org/pdf/1511.06434)
    LR = 2e-4
    B1 = 0.5 # reduces momentum of the gradient (more responsive to fast-changing gradients)
    
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(B1, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=LR, betas=(B1, 0.999))

    # Train GAN (use seed for reproducibility)
    EPOCHS = 150

    torch.manual_seed(MANUALSEED)
    GANresults = train(generator=generator,
                       discriminator=discriminator,
                       trainDataloader=trainDataloader,
                       testDataloader=testDataloader,
                       optimD=optimizerD,
                       optimG=optimizerG,
                       lossFn=GANloss,
                       epochs=EPOCHS,
                       genSamplesPerEpoch=5)
    
    # Save the results and model
    saveGANandResultsMap(generator=generator,
                         discriminator=discriminator,
                         results=GANresults,
                         modelName=f"GAN_CIFAR10_{EPOCHS}_EPOCHS_MODEL.pth",
                         resultsName=f"GAN_CIFAR10_{EPOCHS}_EPOCHS_RESULTS.pth")
    
    # Plot loss curves for GAN
    plotGANLoss(GANresults)
    
    # Plot the generated images from the training loop
    plotGANGeneratorSamples(GANresults, step=10)