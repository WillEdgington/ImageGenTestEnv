import torch

from torch import nn

class UpsampleBlock(nn.Module):
    """Upsample block for generator model.
    """
    def __init__(self, inChannels, outChannels):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    """Generator model for GAN (takes latent space of noise and generates image).
    """
    def __init__(self, latentDim: int=100, # latent vector dimension size
                 imgChannels: int=3, # amount of image channels (default RGB colour channels = 3)
                 featureMapSize: int=64):
        super().__init__()

        # Project latent vector into 4*4 2d space [batchSize, latentDim, 1, 1] -> [batchSize, featureMapSize * 4, 4, 4]
        self.project = nn.Sequential(
            nn.ConvTranspose2d(latentDim, featureMapSize * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(featureMapSize * 4),
            nn.ReLU(True)
        )

        # Upsample 2d space (double the 2d spatial dimensions per block)
        self.upsample = nn.Sequential(
            UpsampleBlock(featureMapSize * 4, featureMapSize * 2), # [batchSize, featureMapSize * 4, 4, 4] -> [batchSize, featureMapSize * 2, 8, 8]
            UpsampleBlock(featureMapSize * 2, featureMapSize) # [batchSize, featureMapSize * 2, 8, 8] -> [batchSize, featureMapSize, 16, 16]
        )

        # Transform 2d space to image with suitable imgChannels and resolution [batchSize, featureMapSize, 16, 16] -> [batchSize, imgChannels, 32, 32]
        self.out = nn.Sequential(
            nn.ConvTranspose2d(featureMapSize, imgChannels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() # want pixel values between [-1, 1]
        )

    def forward(self, x):
        return self.out(self.upsample(self.project(x)))
    
class DownsampleBlock(nn.Module):
    """Downsample block for discriminator model.
    """
    def __init__(self, inChannels: int, outChannels: int, norm: bool=True):
        super().__init__()

        layers = [nn.Conv2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1, bias=False)]

        # Add batch normalization if norm=True
        if norm:
            layers.append(nn.BatchNorm2d(outChannels))
        
        # LeakyReLU the somewhat standard for GAN discriminators
        layers.append(nn.LeakyReLU(0.2, inplace=True)) # LeakyReLU the somewhat standard for 

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    """Discriminator model for GAN (binary classifier for distinguishing real/fake images).
    """
    def __init__(self, imgChannels: int=3, featureMapSize: int=64):
        super().__init__()

        # Block for downsampling the image to extract features
        self.downsample = nn.Sequential(
            DownsampleBlock(imgChannels, featureMapSize, norm=False), # (no BatchNorm in first downsample layer is common practice for stability)
            DownsampleBlock(featureMapSize, featureMapSize * 2),
            DownsampleBlock(featureMapSize * 2, featureMapSize * 4)
        )

        # Classifier head (output single prediction score of probability that image is "real")
        self.classifier = nn.Sequential(
            nn.Conv2d(featureMapSize * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(), # output probabilities of "real"
            nn.Flatten(1) # flatten output
        )

    def forward(self, x):
        return self.classifier(self.downsample(x))