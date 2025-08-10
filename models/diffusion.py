import torch
import math

from torch import nn

# aka: Beta Scheduler
class NoiseScheduler:
    def __init__(self, timesteps):
        self.timesteps = timesteps
    
    def computeBetas(self):
        raise NotImplementedError("Beta computation method must be implemented")
    
    def getAlphas(self) -> tuple[torch.Tensor, torch.Tensor]:
        betas = self.computeBetas()
        alphas = 1.0 - betas
        return alphas, torch.cumprod(alphas, dim=0)
    
    def getNoiseLevel(self, t):
        raise NotImplementedError("Noise level retrieval must be implemented.")
    
class LinearNoiseScheduler(NoiseScheduler):
    def __init__(self, timesteps: int, betaStart: float=1e-4, betaEnd: float=0.02):
        super().__init__(timesteps)
        self.betaStart = betaStart
        self.betaEnd = betaEnd
        self.betas = self.computeBetas()
        self.alphas, self.alphahat = self.getAlphas()

    def computeBetas(self):
        return torch.linspace(self.betaStart, self.betaEnd, self.timesteps)

    def getNoiseLevel(self, t):
        return self.alphahat.to(t.device)[t]
    
class CosineNoiseScheduler(NoiseScheduler):
    def __init__(self, timesteps: int, s: float=8e-3):
        super().__init__(timesteps)
        self.s = s
        self.betas = self.computeBetas()
        self.alphas, self.alphahat = self.getAlphas()

    def computeBetas(self):
        tnorm = torch.linspace(0, self.timesteps, self.timesteps + 1) / self.timesteps
        f = torch.cos((tnorm + self.s) / (1 + self.s) * torch.pi / 2) ** 2
        ahat = f / f[0]
        betas = 1 - (ahat[1:] / ahat[:-1])
        return torch.clip(betas, 1e-4, 0.9999)
    
    def getNoiseLevel(self, t):
        return self.alphahat.to(t.device)[t]
    
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        halfDim = self.dim // 2
        exponent = -math.log(10000) / (halfDim - 1)
        freqs = torch.exp(torch.arange(halfDim, device=t.device) * exponent)
        args = torch.unsqueeze(t, 1) * torch.unsqueeze(freqs, 0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return embedding

class ConvBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int):
        numGroups = min(32, max(inChannels // 4, inChannels)) # four normalization channels
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(numGroups, inChannels),
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=3, padding=1),
            nn.SiLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
class TimeMlp(nn.Module):
    def __init__(self, timeEmbDim: int, outChannels: int, linearLayers: int=1):
        super().__init__()
        assert linearLayers >= 1, "Must be at least one linear layer in time MLP"
        layers = nn.ModuleList()
        layers.append(nn.Linear(timeEmbDim, outChannels))
        layers.append(nn.SiLU())

        for _ in range(linearLayers-1):
            layers.append(nn.Linear(outChannels, outChannels))
            layers.append(nn.SiLU())

        self.timeMlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.timeMlp(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, numHeads: int=1, dropout: float=0):
        super().__init__()
        assert numHeads >= 1, "Must have atleast one attention head in MSA block"

        self.ln = nn.LayerNorm(normalized_shape=channels)

        self.msa = nn.MultiheadAttention(embed_dim=channels,
                                         num_heads=numHeads,
                                         dropout=dropout,
                                         batch_first=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x.view(b, c, h * w).permute(0, 2, 1) # (B, C, H, W) -> (B, HW, C)
        
        x = self.ln(residual)
        x = self.msa(query=x, key=x, value=x)[0] # dont need attention weights

        x = residual + x

        return x.permute(0, 2, 1).view(b, c, h, w) # (B, HW, C) -> (B, C, H, W)
class ResidualBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, timeEmbDim: int, msaHeads: int=0, msaDropout: float=0):
        super().__init__()

        self.block1 = ConvBlock(inChannels=inChannels, outChannels=outChannels)
        self.timeMlp = TimeMlp(timeEmbDim=timeEmbDim, outChannels=outChannels)
        self.block2 = ConvBlock(inChannels=outChannels, outChannels=outChannels)

        self.residualConv = (nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1)
                             if inChannels != outChannels else nn.Identity()
                             )
        
        self.useAttention = bool(msaHeads > 0)
        if msaHeads:
            self.msaBlock = AttentionBlock(channels=outChannels, numHeads=msaHeads, dropout=msaDropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h += self.timeMlp(t).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)

        if self.useAttention:
            h = self.msaBlock(h)
        return h + self.residualConv(x)

class UNet(nn.Module):
    def __init__(self, imgInChannels: int=3, imgOutChannnels: int=3, timeEmbDim: int | None=None, depth: int=3,
                 baseChannels: int=64, numEncHeads: int=0, numDecHeads: int=0, numBotHeads: int=0,
                 encHeadsDropout: float=0, decHeadsDropout: float=0, botHeadsDropout: float=0):
        super().__init__()
        assert depth >= 1, "Depth of encoder/decoder blocks must be at least 1"

        if timeEmbDim is None:
            timeEmbDim = baseChannels * 4

        # Time embedding into an Mlp block
        self.timeMlp = nn.Sequential(
            SinusoidalTimeEmbedding(baseChannels),
            TimeMlp(baseChannels, timeEmbDim, linearLayers=2)
        )

        # Encoder blocks
        self.encoder = nn.ModuleList()
        inChannel = imgInChannels
        for i in range(depth):
            outChannel = baseChannels << i
            self.encoder.append(
                ResidualBlock(inChannels=inChannel, outChannels=outChannel, timeEmbDim=timeEmbDim,
                                  msaHeads=numEncHeads, msaDropout=encHeadsDropout)
            )
            inChannel = outChannel

        # Use avg pooling for downsampling
        self.downsample = nn.AvgPool2d(kernel_size=2)

        # Bottleneck block
        self.bottle = ResidualBlock(inChannels=inChannel, outChannels=inChannel, timeEmbDim=timeEmbDim,
                                    msaHeads=numBotHeads, msaDropout=botHeadsDropout)

        # Decoder blocks
        self.decoder = nn.ModuleList()
        inChannel = baseChannels << (depth - 1)
        for i in range(depth - 1, -1, -1):
            outChannel = baseChannels << max(0, i-1)
            self.decoder.append(
                ResidualBlock(inChannels=inChannel, outChannels=outChannel, timeEmbDim=timeEmbDim,
                              msaHeads=numDecHeads, msaDropout=decHeadsDropout)
            )
            inChannel = outChannel
        
        # Upsample block
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.final = nn.Conv2d(baseChannels, imgOutChannnels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        timeEmb = self.timeMlp(t)

        # Encode
        residuals = []
        for enc in self.encoder:
            x = enc(x, timeEmb)
            residuals.append(x)
            x = self.downsample(x)

        # Bottleneck
        x = self.bottle(x, timeEmb)

        # Decode
        for dec, res in zip(self.decoder, residuals[::-1]):
            x = self.upsample(x) + res
            x = dec(x, timeEmb)

        return self.final(x)