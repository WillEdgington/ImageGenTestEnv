import torch

from utils.save import loadModel, loadResultsMap
from models.gan import Generator, Discriminator
from utils.visualize import plotGANGeneratorSamples
from utils.losses import plotGANLoss, plotGANTrainLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
MANUALSEED = 42

if __name__=="__main__":
    results = loadResultsMap(resultsName="GAN_CIFAR10_150_EPOCHS_RESULTS.pth")
    plotGANGeneratorSamples(results, step=10)
    plotGANTrainLoss(results)
