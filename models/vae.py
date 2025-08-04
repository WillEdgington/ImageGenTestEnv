import torch

from torch import nn

class DownsampleBlock(nn.Module):
    """Downsample block for VAE encoder.
    """
    def __init__(self, inChannels: int, outChannels: int, norm: bool=True, 
                 addConv: int=0, convNorm: bool=False,
                 downAct: nn.ReLU | nn.LeakyReLU | nn.SiLU | nn.GELU=nn.LeakyReLU(0.2, inplace=True),
                 convAct: nn.ReLU | nn.LeakyReLU | nn.SiLU | nn.GELU=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()

        downLayers = nn.ModuleList([nn.Conv2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1, bias=False)])

        # Add batch normalization if norm=True
        if norm:
            downLayers.append(nn.BatchNorm2d(outChannels))
        
        downLayers.append(downAct)
        self.downsample = nn.Sequential(*downLayers)

        self.residual = False
        if addConv:
            self.residual = True
            convLayers = nn.ModuleList()
            for _ in range(addConv):
                convLayers.append(nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False))
                if convNorm:
                    convLayers.append(nn.BatchNorm2d(outChannels))
                convLayers.append(convAct)
            self.conv = nn.Sequential(*convLayers)

    def forward(self, x):
        down = self.downsample(x)
        return down if not self.residual else down + self.conv(down)
    
class UpsampleBlock(nn.Module):
    """Upsample block for VAE decoder
    """
    def __init__(self, inChannels, outChannels, norm: bool=True,
                 addConv: int=0, convNorm: bool=False, 
                 upAct: nn.ReLU | nn.LeakyReLU | nn.SiLU | nn.GELU=nn.ReLU(inplace=True),
                 convAct: nn.ReLU | nn.LeakyReLU | nn.SiLU | nn.GELU=nn.ReLU(inplace=True)):
        super().__init__()

        upLayers = nn.ModuleList([nn.ConvTranspose2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1)])

        if norm:
            upLayers.append(nn.BatchNorm2d(outChannels))
        
        upLayers.append(upAct)
        self.upsample = nn.Sequential(*upLayers)

        self.residual = False
        if addConv:
            self.residual = True
            convLayers = nn.ModuleList()
            for _ in range(addConv):
                convLayers.append(nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False))
                if convNorm:
                    convLayers.append(nn.BatchNorm2d(outChannels))
                convLayers.append(convAct)
            self.conv = nn.Sequential(*convLayers)

    def forward(self, x):
        up = self.upsample(x)
        return up if not self.residual else up + self.conv(up)

class VAE(nn.Module):
    """Variation auto encoder model for image generation
    """
    def __init__(self, latentDim: int=100, imgChannels: int=3, imgDims: int=32, 
                 upAddConv: int=0, downAddConv: int=0, downConvNorm: bool=False, upConvNorm:bool=False,
                 upAct: nn.ReLU | nn.LeakyReLU | nn.SiLU | nn.GELU=nn.ReLU(inplace=True),
                 upExtraConvAct: nn.ReLU | nn.LeakyReLU | nn.SiLU | nn.GELU=nn.ReLU(inplace=True),
                 downAct: nn.ReLU | nn.LeakyReLU | nn.SiLU | nn.GELU=nn.LeakyReLU(0.2, inplace=True),
                 downExtraconvAct: nn.ReLU | nn.LeakyReLU | nn.SiLU | nn.GELU=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.latentDim = latentDim
        self.imgChannels = imgChannels

        self.encoder = nn.Sequential(
            DownsampleBlock(inChannels=imgChannels, outChannels=imgDims, addConv=downAddConv, convNorm=downConvNorm, downAct=downAct, convAct=downExtraconvAct),
            DownsampleBlock(inChannels=imgDims, outChannels=imgDims << 1, addConv=downAddConv, convNorm=downConvNorm, downAct=downAct, convAct=downExtraconvAct),
            DownsampleBlock(inChannels=imgDims << 1, outChannels=latentDim, addConv=downAddConv, convNorm=downConvNorm, downAct=downAct, convAct=downExtraconvAct)
        )

        self.downsampledDims = imgDims >> 3 # imgDims >> n is for n downsampling
        flatFeatSize = latentDim * self.downsampledDims ** 2 # size of flattened encoder output
        self.flatten = nn.Flatten()
        
        self.mu = nn.Linear(in_features=flatFeatSize, out_features=latentDim)
        self.logvar = nn.Linear(in_features=flatFeatSize, out_features=latentDim)

        self.decoderInput = nn.Linear(latentDim, flatFeatSize)
        self.decoder = nn.Sequential(
            UpsampleBlock(inChannels=latentDim, outChannels=latentDim >> 1, addConv=upAddConv, convNorm=upConvNorm, upAct=upAct, convAct=upExtraConvAct),
            UpsampleBlock(inChannels=latentDim >> 1, outChannels=latentDim >> 2, addConv=upAddConv, convNorm=upConvNorm, upAct=upAct, convAct=upExtraConvAct),
            nn.ConvTranspose2d(latentDim >> 2, imgChannels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Want pixel values between [-1, 1]
        )

    # encode x -> μ, ln(σ^2)
    def encode(self, x):
        x = self.flatten(self.encoder(x))
        return self.mu(x), self.logvar(x)

    # z = μ + σ ⊙ ε
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # convert natural log of variance to standard deviation
        eps = torch.randn_like(std) # sample from normal distribution
        return mu + eps * std # z = mu + std ⊙ eps (⊙: element wise multiplication)
        
    # decode z -> x hat
    def decode(self, z):
        x = self.decoderInput(z)
        x = x.view(-1, self.latentDim, self.downsampledDims, self.downsampledDims)
        return self.decoder(x)

    # x -> μ, ln(σ^2) -> z -> x hat, μ, ln(σ^2)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        
def vaeLoss(xhat, x, mu, logvar, beta=1.0):
    reconLoss = nn.functional.mse_loss(xhat, x, reduction='sum')
    klDiv = torch.sum(logvar.exp() + mu.pow(2) - (1 + logvar)) * 0.5
    return reconLoss + beta * klDiv, reconLoss, klDiv

class BetaScheduler:
    def __init__(self, betaInit: float):
        self.beta = betaInit
    def update(self, Dkl: float, recon: float) -> float:
        raise NotImplementedError("BetaScheduler.update() is not implemented. Please implement method.")

class AdaptiveMomentBetaScheduler(BetaScheduler):
    def __init__(self, 
                 betaInit: float=1.0, # initial beta value
                 gamma: float=0.2, # momentum value 
                 eta: float=1.0, # how agressively beta is updated
                 epsilon: float=1e-8, # arbitrarily small value > 0
                 betaMin: float=0.1, # min for beta 
                 betaMax: float=10.0 # max for beta
                 ):
        assert 0 < gamma <= 1, f"momentum value (gamma) must be in range 0 < gamma <= 1. Current value: gamma = {gamma}"
        assert betaMin < betaMax, f"beta min must be less than beta max. Current values: Bmin = {betaMin}, Bmax = {betaMax}"
        assert epsilon > 0, f"epsilon must be an arbitrarily small value greater than zero. Current value: epsilon = {epsilon}"
        super().__init__(betaInit)
        self.gamma = gamma
        self.epsilon = epsilon
        self.eta = eta
        self.betaMin = betaMin
        self.betaMax = betaMax
        
        self.momentumRatio = 1.0

        self.Dkl = None
        self.recon = None

    def update(self, Dkl: float, recon: float) -> float:
        if self.Dkl is None or self.recon is None:
            self.Dkl = Dkl
            self.recon = recon
            return self.beta
        
        # Compute deltas of Dkl loss and reconstruction loss
        DklGrad = 1 + (Dkl - self.Dkl) / (self.Dkl + self.epsilon)
        reconGrad = 1 + (recon - self.recon) / (self.recon + self.epsilon)

        # Compute ratio: ||∇Dkl|| / (||∇RL|| + epsilon)
        ratio = DklGrad / (reconGrad + self.epsilon)
        ratio = max(1e-3, min(ratio, 1e3))

        # Update the momentum: momentumRatio * γ + (1 - γ) * ratio
        self.momentumRatio = (self.momentumRatio * self.gamma) + ((1 - self.gamma) * ratio)

        # Update beta: β = β0 * momentumRatio ** η
        self.beta = self.beta * pow(self.momentumRatio, self.eta)

        # Clip beta in range
        self.beta = max(self.betaMin, min(self.beta, self.betaMax))

        return self.beta
