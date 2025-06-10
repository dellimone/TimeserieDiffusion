import torch
import torch.nn as nn

class TransformerDenoiser(nn.Module):
    """Transformer denoiser"""

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor):
        pass