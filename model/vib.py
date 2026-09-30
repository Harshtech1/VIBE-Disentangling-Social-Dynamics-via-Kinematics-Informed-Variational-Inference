"""Variational latent encoders used by the active VIBE-GECV reconstruction.

Pipeline role:
    Local DINOv2 features [B,K,T,768] are flattened by ``vibe_model.py`` and
    sent here as [B*K*T,768]. This module returns affective and environmental
    latent representations, each with ``latent_dim`` channels.

Scientific note:
    The two branches support the reconstruction's affect/context separation.
    The loss currently applies KL regularization only to the affect branch;
    that documented fidelity deviation is handled in ``loss_function.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import numpy as np

class CausalVIB(nn.Module):
    """Encode one feature vector into affective and environmental latents.

    Inputs:
        x: [..., input_dim], normally flattened local features.
    Outputs:
        Dictionary containing ``mu_*``, ``logvar_*``, and sampled/deterministic
        ``z_*`` tensors, all [..., latent_dim].
    """
    def __init__(self, input_dim=768, latent_dim=256, dropout=0.2):
        super().__init__()
        self.aff_encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, latent_dim * 2))
        self.env_encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, latent_dim * 2))
    def reparameterize(self, mu, logvar):
        """Sample ``z = mu + eps * sigma`` during training; use ``mu`` at eval.

        Writing sampling this way keeps the random term independent of the
        learnable mean/variance, so gradients can still update the encoders.
        Deterministic evaluation avoids adding random prediction variation.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    def forward(self, x):
        """Return both VIB branches for the provided feature vectors."""
        mu_aff, logvar_aff = self.aff_encoder(x).chunk(2, dim=-1)
        mu_env, logvar_env = self.env_encoder(x).chunk(2, dim=-1)
        z_aff = self.reparameterize(mu_aff, logvar_aff)
        z_env = self.reparameterize(mu_env, logvar_env)
        return {'z_aff': z_aff, 'mu_aff': mu_aff, 'logvar_aff': logvar_aff, 'z_env': z_env, 'mu_env': mu_env, 'logvar_env': logvar_env}
