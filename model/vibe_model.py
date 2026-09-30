"""Top-level VIBE-GECV reconstruction model.

Pipeline: local [B,K,T,768] + audio [B,T,768] + global scene [B,768] + gamma
[B,1,1] -> VIB -> context/audio interaction -> temporal transformer -> pooled
clip representation -> three logits (Positive=0, Neutral=1, Negative=2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import numpy as np

from .vib import CausalVIB
from .context_gate import ContextInjectionGate
from .learned_attenttion import LearnedQueryAttention, CrossModalAttentionBlock
from .gamma_transformer import DecoupledAdaLN, GammaGatedTransformer


class VIBE_Transformer(nn.Module):
    """Assemble the active reconstruction's VIB, attention, and classifier path."""
    def __init__(self, input_dim=768, latent_dim=256, num_classes=3, attention_heads=8,
                 transformer_depth=2, dropout=None, drop_path=0.1,
                 enable_context_injection=True, gamma_conditioning='stored',
                 enable_cross_modal_attention=True, audio_input_mode='observed',
                 global_scene_conditioning='observed'):
        super().__init__()
        if gamma_conditioning not in {'stored', 'fixed_zero'}:
            raise ValueError(f"Unknown gamma_conditioning: {gamma_conditioning}")
        if audio_input_mode not in {'observed', 'zero_projected_tokens'}:
            raise ValueError(f"Unknown audio_input_mode: {audio_input_mode}")
        if global_scene_conditioning not in {'observed', 'fixed_neutral'}:
            raise ValueError(f"Unknown global_scene_conditioning: {global_scene_conditioning}")
        self.enable_context_injection = bool(enable_context_injection)
        self.gamma_conditioning = gamma_conditioning
        self.enable_cross_modal_attention = bool(enable_cross_modal_attention)
        self.audio_input_mode = audio_input_mode
        self.global_scene_conditioning = global_scene_conditioning
        # Existing topology is unchanged.  ``dropout=None`` preserves each
        # legacy component default; a configured value applies one documented
        # reconstruction dropout rate to the already-existing dropout layers.
        vib_dropout = 0.2 if dropout is None else dropout
        context_dropout = 0.1 if dropout is None else dropout
        attention_dropout = 0.1 if dropout is None else dropout
        classifier_dropout = 0.5 if dropout is None else dropout
        # --- Stage 1: Global Pre-processing ---
        self.spatial_pool = LearnedQueryAttention(input_dim=input_dim, hidden_dim=128)
        
        # Style Projector
        self.style_dim = 64
        self.global_style_proj = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, self.style_dim) 
        )
        self.scene_scaler = nn.Sequential(nn.Linear(self.style_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

        # --- Stage 2: Disentanglement ---
        self.vib = CausalVIB(input_dim, latent_dim, dropout=vib_dropout)
        self.context_gate = ContextInjectionGate(latent_dim, dropout=context_dropout) # Local-only
        self.audio_proj = nn.Linear(input_dim, latent_dim)
        
        # --- Stage 3: Interaction Reasoning ---
        self.cross_modal_block = CrossModalAttentionBlock(latent_dim, heads=attention_heads, dropout=attention_dropout) # Vis-Aud only
        
        # NEW: Transformer with Decoupled Modulation
        self.temporal_encoder = GammaGatedTransformer(
            dim=latent_dim, 
            style_dim=self.style_dim, 
            depth=transformer_depth, heads=attention_heads, dropout=attention_dropout, drop_path=drop_path
        )
        
        # --- Stage 4: Pooling & Classify ---
        self.temporal_pool = LearnedQueryAttention(input_dim=latent_dim, hidden_dim=128)
        self.contrastive_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 128))
        
        fusion_dim = latent_dim + 1 
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024), 
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout(classifier_dropout),
            nn.Linear(1024, num_classes)
        )
        self.cot_aligner = nn.Linear(latent_dim, input_dim)

    def forward_with_gamma(self, local_p, global_v, env_feat, audio_seq, gamma, mixup_alpha=None, targets=None,
                           agent_mask=None, local_time_mask=None, audio_time_mask=None, global_time_mask=None,
                           reconstruction_mode=False):
        """Compute class logits from a padded multimodal reconstruction batch.

        Shapes: local_p [B,K,T,768], audio_seq [B,T,768], global_v [B,768],
        gamma [B,1,1], local_time_mask [B,K,T], and audio_time_mask [B,T].
        ``env_feat`` is a legacy API argument and has no current read site.
        """
        # env_feat is retained for legacy-call compatibility.  The current model
        # never consumes it.  VIBE-GECV-RECON-v1.0 therefore passes None rather
        # than inventing an environment tensor from global_scene.
        B, K, T, D = local_p.shape
        if agent_mask is None:
            agent_mask = torch.ones(B, K, dtype=torch.bool, device=local_p.device)
        else:
            agent_mask = agent_mask.to(device=local_p.device, dtype=torch.bool)
        if local_time_mask is None:
            local_time_mask = agent_mask.unsqueeze(-1).expand(-1, -1, T)
        else:
            local_time_mask = local_time_mask.to(device=local_p.device, dtype=torch.bool)
        # Make agent masking explicit even if a caller supplies an inconsistent
        # time mask. This is identity-preserving for the frozen K=8 baseline.
        local_time_mask = local_time_mask & agent_mask.unsqueeze(-1)
        if audio_time_mask is None:
            audio_time_mask = torch.ones(B, audio_seq.shape[1], dtype=torch.bool, device=local_p.device)
        
        # 1. Spatial Pooling for Global
        if global_v.dim() == 2:
            global_feat_avg = global_v
        elif global_v.dim() == 3 and global_v.shape[1] > T:
            if reconstruction_mode:
                raise ValueError("VIBE-GECV-RECON-v1.0 forbids global token-to-T reshaping; pass global_scene [B,768].")
            spatial_patches = global_v.shape[1] // T
            global_v_spatial = global_v.view(B * T, spatial_patches, D)
            pooled_spatial, _ = self.spatial_pool(global_v_spatial)
            global_v = pooled_spatial.view(B, T, D)
            global_feat_avg = global_v.mean(dim=1)
        else:
            if global_time_mask is None:
                global_feat_avg = global_v.mean(dim=1)
            else:
                weights = global_time_mask.unsqueeze(-1).to(global_v.dtype)
                global_feat_avg = (global_v * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        # 2. Compute Global Style Token.  Fixed-neutral bypasses only the
        # data-derived scene pathway while retaining expected dimensions.
        if self.global_scene_conditioning == 'observed':
            style_token = self.global_style_proj(global_feat_avg) # [B, style_dim]
            # ------------------------------------------------------------------
            # V1.1 CORRECTION — SAMPLE-WISE GAMMA / SCENE-SCALE BROADCASTING
            # Historical V1.0 used incompatible singleton dimensions. PyTorch
            # could broadcast gamma [B,1,1] and scene scale [B,1] to [B,B,1],
            # allowing one batch item's conditioning to affect another's.
            # Keep both tensors [B,1,1] so every batch item keeps one scalar.
            # This is a reconstruction implementation issue, not a paper claim.
            # Regression guard: tests/gamma_scene_scale_invariance_cpu.py
            # ------------------------------------------------------------------
            # Keep a time/scalar axis: [B,1,1] prevents unintended [B,B,1]
            # cross-sample broadcasting when combined with gamma [B,1,1].
            scene_scale = self.scene_scaler(style_token).unsqueeze(-1)
        else:
            style_token = global_feat_avg.new_zeros(B, self.style_dim)
            scene_scale = global_feat_avg.new_ones(B, 1, 1)
        
        # 3. Disentangle
        vib_out = self.vib(local_p.view(-1, D))
        z_aff = vib_out['z_aff'].view(B, K, T, -1)
        z_env = vib_out['z_env'].view(B, K, T, -1)
        local_mask = local_time_mask.unsqueeze(-1).to(z_aff.dtype)
        z_aff = z_aff * local_mask
        z_env = z_env * local_mask
        for key in ('mu_aff', 'logvar_aff', 'z_aff', 'z_env'):
            vib_out[key] = vib_out[key].view(B, K, T, -1) * local_mask
        
        # 4. Mixup
        if self.training and mixup_alpha is not None and targets is not None:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(B).to(local_p.device)
            z_aff = lam * z_aff + (1 - lam) * z_aff[index]
            z_env = lam * z_env + (1 - lam) * z_env[index]
            style_token = lam * style_token + (1 - lam) * style_token[index] 
            gamma = lam * gamma + (1 - lam) * gamma[index]
            scene_scale = lam * scene_scale + (1 - lam) * scene_scale[index]
            targets_a, targets_b = targets, targets[index]
            mixup_data = (targets_a, targets_b, lam)
        else:
            mixup_data = None

        # 5. Local Context Gating
        if self.enable_context_injection:
            z_aff_context = self.context_gate(z_aff, z_env) * local_mask
        else:
            # Removing injection must not also remove the existing LayerNorm.
            z_aff_context = self.context_gate.norm(z_aff) * local_mask
        
        # 6. Cross-Modal Attention
        vis_tokens = z_aff_context.permute(0, 2, 1, 3).reshape(B, T*K, -1)
        aud_tokens = self.audio_proj(audio_seq)
        if self.audio_input_mode == 'zero_projected_tokens':
            aud_tokens = torch.zeros_like(aud_tokens)
        vis_mask = local_time_mask.permute(0, 2, 1).reshape(B, T * K)
        refined_vis = self.cross_modal_block(
            vis_tokens, aud_tokens, vis_mask=vis_mask, audio_mask=audio_time_mask,
            enable_cross_modal_attention=self.enable_cross_modal_attention,
        )
        
        # 7. Global-Adaptive Temporal Encoding
        refined_vis = refined_vis.view(B, T, K, -1)
        agent_weights = local_time_mask.permute(0, 2, 1).unsqueeze(-1).to(refined_vis.dtype)
        temporal_mask = local_time_mask.any(dim=1)
        refined_vis = (refined_vis * agent_weights).sum(dim=2) / agent_weights.sum(dim=2).clamp_min(1.0)
        if self.gamma_conditioning == 'fixed_zero':
            gamma = torch.zeros_like(gamma)
        # Both factors are [B,1,1] in V1.1: synchrony remains sample-wise.
        gamma_prime = gamma * scene_scale 
        
        # Pass Global Style to DecoupledAdaLN
        h_seq = self.temporal_encoder(refined_vis, gamma_prime, style_token, time_mask=temporal_mask)
        
        # 8. Output
        h_final, attn_weights = self.temporal_pool(h_seq, mask=temporal_mask)
        proj_feat = self.contrastive_head(h_final)
        rationale_proj = self.cot_aligner(h_final)
        
        # Handle mismatch in gamma for final fusion as well
        if gamma_prime.shape[1] != h_final.shape[1] and gamma_prime.shape[1] > 1:
             # Just pick the last value or mean if shapes don't align for fusion (usually not an issue for classification head input)
             # But for safety, we usually fuse a global scalar summary of gamma
             pass
        
        # For fusion, we just take the mean gamma over time to get a single scalar per video
        gamma_scalar = gamma_prime.mean(dim=1) 
        
        # Concatenate pooled latent [B,L] and one gamma summary [B,1].
        fusion_input = torch.cat([h_final, gamma_scalar], dim=-1)
        # Logits are unnormalized scores [B,3]; metric code maps indices to
        # Positive=0, Neutral=1, Negative=2 and applies argmax at evaluation.
        logits = self.classifier(fusion_input)
        
        return {
            'logits': logits, 'attn_weights': attn_weights, 'features': proj_feat, 
            'h_final': h_final, 'vib_out': vib_out, 'rationale': rationale_proj, 'mixup_data': mixup_data
        }
