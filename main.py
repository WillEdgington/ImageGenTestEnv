import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
    dataloader = prepareData()



