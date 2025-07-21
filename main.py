import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary

from models.gan import Generator, Discriminator

device = "cuda" if torch.cuda.is_available() else "cpu"

def collectData(train: bool=True):
    # Create transform pipeline for images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scales to [-1,1] (Common for the Gen models in use here)
    ])

    # Collect CIFAR10 training dataset with applied transform pipeline
    dataset = datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
    return dataset

def createDataLoader(dataset: datasets.CIFAR10, batchSize: int=32, numWorkers: int=2, shuffle: bool=True):
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=numWorkers)
    return dataloader

def prepareData(train: bool=True, batchSize: int=32, numWorkers: int=2):
    dataset = collectData(train=train)
    dataloader = createDataLoader(dataset, batchSize=batchSize, numWorkers=numWorkers, shuffle=train)
    return dataloader

if __name__=="__main__":
    # Prepare data loaders
    dataloader = prepareData()

    # Create instance of generator for GAN model
    generator = Generator(latentDim=100, imgChannels=3, featureMapSize=64)
    generator.to(device) # send to chosen device (GPU if possible)

    # # Get a summary of generator (uncomment to see)
    # summary(generator, 
    #         input_size=(1, 100, 1, 1),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])

    # Create instance of discriminator for GAN model
    discriminator = Discriminator(imgChannels=3, featureMapSize=64)
    discriminator.to(device)

    # # Get a summary of discriminator (uncomment to see)
    # summary(discriminator, 
    #         input_size=(1, 3, 32, 32),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])




