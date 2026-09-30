"""Attention utilities for pooling and visual/audio interaction in VIBE-GECV.

``LearnedQueryAttention`` reduces a token sequence to one vector. The
cross-modal block lets visual person/time tokens query audio tokens before
visual self-attention refines the result.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import numpy as np

# ==========================================
# 1. ATTENTION POOLING
# ==========================================
class LearnedQueryAttention(nn.Module):
    """Learn a query that weights valid sequence tokens during pooling."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.attn_proj = nn.Linear(input_dim, hidden_dim)
        self.query_vector = nn.Parameter(torch.randn(hidden_dim, 1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        """Pool [B,T,D] or [B,N,D] into [B,D]; mask marks valid positions."""
        u = torch.tanh(self.attn_proj(x)) 
        scores = torch.matmul(u, self.query_vector).squeeze(-1) 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1) 
        context_vector = torch.sum(x * attn_weights.unsqueeze(-1), dim=1) 
        return context_vector, attn_weights


class CrossModalAttentionBlock(nn.Module):
    """Fuse visual queries [B,T*K,L] with audio keys/values [B,T,L]."""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.dropout = nn.Dropout(dropout)
    def forward(self, vis, aud, vis_mask=None, audio_mask=None, enable_cross_modal_attention=True):
        """Return a masked visual sequence with unchanged [B,T*K,L] shape.

        The optional false branch is an ablation-only intervention. It removes
        audio-to-visual attention but deliberately retains normalization,
        visual self-attention, FFN processing, and mask behavior.
        """
        kv = aud
        q = vis
        audio_padding = None if audio_mask is None else ~audio_mask.bool()
        vis_padding = None if vis_mask is None else ~vis_mask.bool()
        if enable_cross_modal_attention:
            attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=audio_padding)
            vis = self.norm1(vis + self.dropout(attn_out))
        else:
            # Preserve the existing normalization, self-attention, FFN, masks,
            # and output width while removing only audio-to-visual attention.
            vis = self.norm1(vis)
        attn_out, _ = self.self_attn(vis, vis, vis, key_padding_mask=vis_padding)
        vis = self.norm2(vis + self.dropout(attn_out))
        vis = self.norm3(vis + self.ffn(vis))
        if vis_mask is not None:
            vis = vis * vis_mask.unsqueeze(-1).to(vis.dtype)
        return vis
