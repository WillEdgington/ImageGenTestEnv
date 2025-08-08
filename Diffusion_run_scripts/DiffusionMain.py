import torch
import sys

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.diffusion import UNet, ConvBlock, ResidualBlock, AttentionBlock, TimeMlp

device = "cuda" if torch.cuda.is_available() else "cpu"
MANUALSEED = 42

BASECHANNELS = 64
IMGCHANNELS = 3
TIMEEMBDIM = None
DEPTH = 3
ENCHEADS = 4
DECHEADS = 4
BOTHEADS = 4
ENCHEADDROP = 0
DECHEADDROP = 0
BOTHEADDROP = 0


if __name__=="__main__":
    unet = UNet(imgInChannels=IMGCHANNELS,
                imgOutChannnels=IMGCHANNELS,
                timeEmbDim=TIMEEMBDIM,
                depth=DEPTH,
                baseChannels=BASECHANNELS,
                numEncHeads=ENCHEADS,
                numDecHeads=DECHEADS,
                numBotHeads=BOTHEADS,
                encHeadsDropout=ENCHEADDROP,
                decHeadsDropout=DECHEADDROP,
                botHeadsDropout=BOTHEADDROP)

    summary(model=unet,
            input_size=[(1, 3, 32, 32),(1,)],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])