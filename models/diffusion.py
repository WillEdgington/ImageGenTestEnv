import torch
import math

from torch import nn
from typing import List, Tuple

from models.LDM import LDMVAE

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

class ResidualCluster(nn.Module):
    def __init__(self, numBlocks: int, inChannels: int, outChannels: int, timeEmbDim: int, msaHeads: int=0, msaDropout: float=0):
        super().__init__()
        assert numBlocks >= 1, "Number of ResidualBlock in a ResidualCluster must be greater than or equal to one."
        self.cluster = nn.ModuleList([
            ResidualBlock(inChannels=(inChannels if b==0 else outChannels), outChannels=outChannels, timeEmbDim=timeEmbDim,
                          msaHeads=msaHeads, msaDropout=msaDropout) for b in range(numBlocks)
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for res in self.cluster:
            x = res(x, t)
        return x

class UNet(nn.Module):
    def __init__(self, imgInChannels: int=3, imgOutChannnels: int=3, timeEmbDim: int | None=None, depth: int=3,
                 resBlocks: Tuple[int,int,int]|int=1, baseChannels: int=64, numEncHeads: int=0, numDecHeads: int=0, numBotHeads: int=0,
                 encHeadsDropout: float=0, decHeadsDropout: float=0, botHeadsDropout: float=0):
        super().__init__()
        assert depth >= 1, "Depth of encoder/decoder blocks must be at least 1"

        if timeEmbDim is None:
            timeEmbDim = baseChannels * 4

        if resBlocks is not None and isinstance(resBlocks, int):
            resBlocks = (resBlocks, resBlocks, resBlocks)

        assert min(resBlocks) >= 1, "Must have one ResidualBlock per encoder/bottleneck/decoder block."

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
            if resBlocks[0] == 1:
                self.encoder.append(
                    ResidualBlock(inChannels=inChannel, outChannels=outChannel, timeEmbDim=timeEmbDim,
                                  msaHeads=numEncHeads, msaDropout=encHeadsDropout)
                )
            else:
                self.encoder.append(
                    ResidualCluster(numBlocks=resBlocks[0], inChannels=inChannel, outChannels=outChannel, timeEmbDim=timeEmbDim,
                                    msaHeads=numEncHeads, msaDropout=encHeadsDropout)
                )

            inChannel = outChannel

        # Use avg pooling for downsampling
        self.downsample = nn.AvgPool2d(kernel_size=2)

        # Bottleneck block
        if resBlocks[1] == 1:
            self.bottle = ResidualBlock(inChannels=inChannel, outChannels=inChannel, timeEmbDim=timeEmbDim,
                                        msaHeads=numBotHeads, msaDropout=botHeadsDropout)
        else:
            self.bottle = ResidualCluster(numBlocks=resBlocks[1], inChannels=inChannel, outChannels=inChannel, timeEmbDim=timeEmbDim,
                                          msaHeads=numBotHeads, msaDropout=botHeadsDropout)

        # Decoder blocks
        self.decoder = nn.ModuleList()
        inChannel = baseChannels << (depth - 1)
        for i in range(depth - 1, -1, -1):
            outChannel = baseChannels << max(0, i-1)
            if resBlocks[2] == 1:
                self.decoder.append(
                    ResidualBlock(inChannels=inChannel, outChannels=outChannel, timeEmbDim=timeEmbDim,
                                  msaHeads=numDecHeads, msaDropout=decHeadsDropout)
                )
            else:
                self.decoder.append(
                    ResidualCluster(numBlocks=resBlocks[2], inChannels=inChannel, outChannels=outChannel, timeEmbDim=timeEmbDim,
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
    
@torch.no_grad
def sample(model: torch.nn.Module,
           noiseScheduler: NoiseScheduler,
           xT: torch.Tensor,
           autoencoder: LDMVAE|None=None,
           skip: int=1,
           eta: float=1.0,
           getSteps: int | None=None,
           device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> torch.Tensor | List[Tuple[int, torch.Tensor]]:
    timesteps = noiseScheduler.timesteps
    assert skip <= timesteps, f"skip variable must be less than or equal to T. T: {timesteps}"
    assert (getSteps is None) or (getSteps % skip == 0), f"getSteps must be divisible by skip. getSteps: {getSteps}, skip: {skip}"

    xt = xT.to(device)

    if getSteps is not None:
        samples = []
    t = timesteps

    while t > 0:
        if getSteps is not None and (timesteps - t) % getSteps == 0:
            samples.append((t, xt if autoencoder is None else autoencoder.decode(xt).detach()))
        t -= skip
        xt = sampleStep(model=model,
                        noiseScheduler=noiseScheduler,
                        xt=xt,
                        t=t,
                        skip=skip,
                        eta=eta,
                        device=device)
    
    if getSteps is not None:
        samples.append((0, xt if autoencoder is None else autoencoder.decode(xt).detach()))

    return (xt if autoencoder is None else autoencoder.decode(xt).detach()) if getSteps is None else samples

def sampleStep(model: torch.nn.Module,
               noiseScheduler: NoiseScheduler,
               xt: torch.Tensor,
               t: int,
               skip: int=1,
               eta: float=1.0,
               device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
    batchSize = xt.shape[0]
    skip = min(skip, t)

    tBatch = torch.full((batchSize,), t, device=device, dtype=torch.int64)
    skipBatch = torch.full((batchSize,), skip, device=device, dtype=torch.int64)

    alphahatt = noiseScheduler.getNoiseLevel(tBatch).view(batchSize, 1, 1, 1)
    alphahatprev = noiseScheduler.getNoiseLevel(t - skipBatch).view(batchSize, 1, 1, 1)

    eps = model(xt, tBatch)
    x0pred = (xt - torch.sqrt(1 - alphahatt) * eps) / torch.sqrt(alphahatt)

    sigma = eta * torch.sqrt((1 - alphahatprev) / (1 - alphahatt)) * torch.sqrt(1 - alphahatt / alphahatprev)
    mu = torch.sqrt(alphahatprev) * x0pred + torch.sqrt(1 - alphahatprev - sigma**2) * eps
    z = torch.randn_like(xt) if (t - skip) > 0 else torch.zeros_like(xt) # dont add noise to x0

    return (mu + (sigma * z))