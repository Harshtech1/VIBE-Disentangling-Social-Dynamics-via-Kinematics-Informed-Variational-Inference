"""Context injection between VIB affective and environmental representations.

Data flow: ``z_aff`` and ``z_env`` [B,K,T,L] from ``model/vib.py`` -> this
gate -> context-modulated affective tokens for cross-modal attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import numpy as np

class ContextInjectionGate(nn.Module):
    """Use environment/context features to apply an elementwise affective gate."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.context_net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim * 2)
        )
        with torch.no_grad():
            self.context_net[-1].weight.zero_()
            self.context_net[-1].bias.zero_()
    def forward(self, z_aff, z_env):
        """Return LayerNorm(z_aff) * (1 + scale(z_env)) + shift(z_env).

        Shapes are preserved: both inputs and the output are [..., dim]. The
        zero initialization above makes the gate start as normalized affective
        features, leaving learned contextual modulation to training.
        """
        scale, shift = self.context_net(z_env).chunk(2, dim=-1)
        return self.norm(z_aff) * (1 + scale) + shift
