import torch

from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, List

def createTransformsList(imgSize: Tuple[int, int]|int|None=None, normalize: bool=True,
                         augment: float=0):
    if isinstance(imgSize, int):
        imgSize = (imgSize, imgSize)

    transList = [transforms.Resize(imgSize)] if imgSize is not None else []

    if augment > 0:
        augs = [
            transforms.RandomHorizontalFlip(p=0.5 * augment),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2*augment,
                                       contrast=0.2*augment,
                                       saturation=0.2*augment,
                                       hue=0.1*augment)
            ], p=0.5*augment),
            transforms.RandomAffine(degrees=int(augment*30),
                                    translate=(0.1*augment, 0.1*augment),
                                    scale=(1 - 0.2 * augment, 1 + 0.2 * augment))
        ]
        transList += augs

    transList.append(transforms.ToTensor())
    if normalize:
        transList.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transList

# Cifar-10 32x32
def collectCIFAR10Data(train: bool=True, normalize: bool=True, augment: float=0):
    transform = transforms.Compose(createTransformsList(normalize=normalize, 
                                                        augment=augment))
    
    dataset = datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
    return dataset

# StanfordCars 224x224
# Can not download through PyTorch anymore as original link is broken so i used kaggle:
# https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars
def collectStanfordCarsData(train: bool=True, normalize: bool=True, imgSize: int=64, augment: float=0):
    assert imgSize <= 224, f"imgSize ({imgSize}) must be less than or equal to size of min HW in stanford cars images (224)."
    transform = transforms.Compose(createTransformsList(imgSize=imgSize if imgSize!=224 else None,
                                                        normalize=normalize,
                                                        augment=augment))

    dataset = datasets.StanfordCars(root="./data", split="train" if train else "test", transform=transform)

    return dataset

# CelebA 178 x 218
def collectCelebAData(train: bool=True, normalize: bool=True, imgSize: int=64, augment: float=0):
    assert imgSize <= 178, f"imgSize ({imgSize}) must be less than or equal to size of min HW in CelebA images (178)."
    transList = [transforms.CenterCrop(178)] + createTransformsList(imgSize=imgSize if imgSize != 178 else None, normalize=normalize, augment=augment)
    transform = transforms.Compose(transList)

    dataset = datasets.CelebA(root="./data", split="train" if train else "test", download=True, transform=transform)

    return dataset

def createDataLoader(dataset: datasets.CIFAR10, batchSize: int=32, numWorkers: int=2, shuffle: bool=True, seed: int=42):
    torch.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=numWorkers, pin_memory=True)
    return dataloader

def prepareData(data: str="CIFAR10", train: bool=True, batchSize: int=32, numWorkers: int=2, seed: int=42, normalize: bool=True, imgSize: int=64, augment: float=0, device: torch.device="cpu"):
    if not train:
        assert augment == 0, "Do not apply augmentation to validation/test dataset."

    match data:
        case "CIFAR10":
            dataset = collectCIFAR10Data(train=train, normalize=normalize, augment=augment)
        case "STANFORDCARS":
            dataset = collectStanfordCarsData(train=train, normalize=normalize, imgSize=imgSize, augment=augment)
        case "CELEBA":
            dataset = collectCelebAData(train=train, normalize=normalize, imgSize=imgSize, augment=augment)
        case _:
            ValueError(f"Unsupported dataset: {data}")

    return createDataLoader(dataset, batchSize=batchSize, numWorkers=numWorkers, shuffle=train, seed=seed)