import torch
from torch import nn
from typing import Tuple

class ConvBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, numGroups: int|None=None):
        if numGroups is None:
            numGroups = min(32, max(inChannels // 4, inChannels)) # four normalization channels
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(numGroups, inChannels),
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, numConvs: int=2, numGroups: int|None=None):
        super().__init__()
        assert numConvs >= 1, "Must have atleast one convolutional block in residual block"

        layers = nn.ModuleList()
        layers.append(ConvBlock(inChannels=inChannels, outChannels=outChannels, numGroups=numGroups))
        for _ in range(1, numConvs):
            layers.append(ConvBlock(inChannels=outChannels, outChannels=outChannels, numGroups=numGroups))
        
        self.block = nn.Sequential(*layers)
        self.res = (nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1)
                    if inChannels != outChannels else nn.Identity()
                    )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.res(x)
    
class DownsampleBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, bias: bool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=4, stride=2, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
class UpsampleBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, bias: bool=False,
                 mode: str="nearest"):
        super().__init__()
        assert mode in {"nearest","bilinear","bicubic","area"}, "mode must be one of: 'nearest','bilinear','bicubic','area'."
        self.mode = mode
        self.conv = nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=3, padding=1, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.nn.functional.interpolate(x, scale_factor=2.0, mode=self.mode))
    
class Encoder(nn.Module):
    def __init__(self, inChannels: int=3, baseChannels: int=64, latentChannels: int=4,
                 numDown: int=3, resBlocks: int=2, numResConvs: int=2):
        super().__init__()
        layers = nn.ModuleList([
            nn.Conv2d(in_channels=inChannels, out_channels=baseChannels, kernel_size=3, padding=1, bias=False)
            ])
        channels = baseChannels
        for i in range(1,numDown+1):
            layers += [ResidualBlock(inChannels=channels, outChannels=channels, numConvs=numResConvs) for _ in range(resBlocks)]
            layers.append(DownsampleBlock(inChannels=channels, outChannels=baseChannels<<i))
            channels = baseChannels << i
        
        layers += [ResidualBlock(inChannels=channels, outChannels=channels, numConvs=numResConvs) for _ in range(resBlocks)]

        self.block = nn.Sequential(*layers)
        self.convMu = nn.Conv2d(channels, latentChannels, kernel_size=3, padding=1)
        self.convLogvar = nn.Conv2d(channels, latentChannels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.block(x)
        return self.convMu(x), self.convLogvar(x)
    
class Decoder(nn.Module):
    def __init__(self, outChannels: int=3, baseChannels: int=64, latentChannels: int=4, 
                 numDown: int=3, resBlocks: int=2, numResConvs: int=2):
        super().__init__()
        channels = baseChannels << numDown
        layers = nn.ModuleList([
            nn.Conv2d(in_channels=latentChannels, out_channels=channels, kernel_size=3, padding=1, bias=False)
            ])
        layers += [ResidualBlock(inChannels=channels, outChannels=channels, numConvs=numResConvs) for _ in range(resBlocks)]

        for i in range(numDown - 2, -1, -1):
            layers.append(UpsampleBlock(inChannels=channels, outChannels=baseChannels<<i))
            channels = baseChannels << i
            layers += [ResidualBlock(inChannels=channels, outChannels=channels, numConvs=numResConvs) for _ in range(resBlocks)]

        layers += [nn.Conv2d(in_channels=channels, out_channels=outChannels, kernel_size=3, padding=1),
                   nn.Tanh()]
        self.dec = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

class LDMVAE(nn.Module):
    def __init__(self, imgChannels: int=3, baseChannels: int=64, latentChannels: int=4, numDown: int=3,
                 resBlocks: Tuple[int, int] | int=2, numResConvs: Tuple[int, int] | int=2, stochastic: bool=True):
        if isinstance(resBlocks, int):
            resBlocks = (resBlocks, resBlocks)
        if isinstance(numResConvs, int):
            numResConvs = (numResConvs, numResConvs)
        
        super().__init__()
        self.encoder = Encoder(inChannels=imgChannels, baseChannels=baseChannels, latentChannels=latentChannels, 
                               numDown=numDown, resBlocks=resBlocks[0], numResConvs=numResConvs[0])
        self.decoder = Decoder(outChannels=imgChannels, baseChannels=baseChannels, latentChannels=latentChannels, 
                               numDown=numDown, resBlocks=resBlocks[0], numResConvs=numResConvs[0])
        self.latentChannels = latentChannels
        self.stochastic = stochastic
    
    def reparameterize(self, mu, logvar) -> torch.Tensor:
        if not self.stochastic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar