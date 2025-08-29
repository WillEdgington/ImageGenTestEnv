import torch

from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Cifar-10 32x32
def collectCIFAR10Data(train: bool=True, normalize: bool=True):
    transList = [transforms.ToTensor()]
    if normalize:
        transList.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transList)

    dataset = datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)

    return dataset

# StanfordCars 224x224
# Can not download through PyTorch anymore as original link is broken so i used kaggle:
# https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars
def collectStanfordCarsData(train: bool=True, normalize: bool=True, imgSize: int=64):
    assert imgSize <= 224, f"imgSize ({imgSize}) must be less than or equal to size of min HW in stanford cars images (224)."
    transList = [transforms.Resize((imgSize, imgSize)), transforms.ToTensor()]
    if normalize:
        transList.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transList)

    dataset = datasets.StanfordCars(root="./data", split="train" if train else "test", transform=transform)

    return dataset

# CelebA 178 x 218
def collectCelebAData(train: bool=True, normalize: bool=True, imgSize: int=64):
    assert imgSize <= 178, f"imgSize ({imgSize}) must be less than or equal to size of min HW in CelebA images (178)."
    transList = [transforms.CenterCrop(178), transforms.Resize((imgSize, imgSize)), transforms.ToTensor()]
    if normalize:
        transList.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transList)

    dataset = datasets.CelebA(root="./data", split="train" if train else "test", download=True, transform=transform)

    return dataset

def createDataLoader(dataset: datasets.CIFAR10, batchSize: int=32, numWorkers: int=2, shuffle: bool=True, seed: int=42):
    torch.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=numWorkers, pin_memory=True)
    return dataloader

def prepareData(data: str="CIFAR10", train: bool=True, batchSize: int=32, numWorkers: int=2, seed: int=42, normalize: bool=True, imgSize: int=64, device: torch.device="cpu"):
    match data:
        case "CIFAR10":
            dataset = collectCIFAR10Data(train=train, normalize=normalize)
        case "STANFORDCARS":
            dataset = collectStanfordCarsData(train=train, normalize=normalize, imgSize=imgSize)
        case "CELEBA":
            dataset = collectCelebAData(train=train, normalize=normalize, imgSize=imgSize)
        case _:
            ValueError(f"Unsupported dataset: {data}")

    return createDataLoader(dataset, batchSize=batchSize, numWorkers=numWorkers, shuffle=train, seed=seed)