"""Training objectives for the VIBE-GECV reconstruction.

The active V1.1 objective combines hierarchical classification, affect-branch
KL regularization, affect/environment orthogonality, and optional text SAT.
SAT is disabled in the approved reconstruction because text anchors are absent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalSmoothingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Target: Pos(0), Neu(1), Neg(2)
        self.adj = torch.tensor([
            [0.85, 0.15, 0.00], 
            [0.10, 0.80, 0.10], 
            [0.00, 0.15, 0.85]  
        ])
            
    def forward(self, logits, target_indices):
        self.adj = self.adj.to(logits.device)
        soft_targets = self.adj[target_indices]
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()


class VIBE_Loss(nn.Module):
    """Calculate the documented reconstruction loss terms from a model batch."""
    def __init__(self, lambda_sat=0.5, lambda_ortho=0.1, lambda_kl=1.0, enable_sat=True):
        super().__init__()
        self.hierarchical_ce = HierarchicalSmoothingLoss()
        
        # Hyperparameters for structural regularization
        self.weights = {
            'sat': lambda_sat, 
            'ortho': lambda_ortho,
            # The historical objective added l_kl directly, i.e. with an
            # implicit coefficient of 1.0.  Keep that exact default explicit.
            'kl': lambda_kl,
        }
        self.enable_sat = enable_sat

    def forward(self, output, targets, batch):
        """Return total, HCE, KL, SAT, and orthogonality scalar losses.

        Fidelity note: despite dual VIB branches, the current KL expression
        below uses only affective ``mu_aff``/``logvar_aff``. The paper describes
        KL over both branches; this documentation pass intentionally does not
        change the frozen reconstruction behavior.
        """
        mixup_data = output.get('mixup_data')
        
        # 1. Hierarchical Label Smoothing Loss - L_HCE
        if mixup_data is not None:
            targets_a, targets_b, lam = mixup_data
            loss_a = self.hierarchical_ce(output['logits'], targets_a)
            loss_b = self.hierarchical_ce(output['logits'], targets_b)
            l_hce = lam * loss_a + (1 - lam) * loss_b
        else:
            l_hce = self.hierarchical_ce(output['logits'], targets)
        
        # 2. VIB KL Loss - L_KL
        vib = output['vib_out']
        l_kl = (-0.5 * torch.sum(1 + vib['logvar_aff'] - vib['mu_aff'].pow(2) - vib['logvar_aff'].exp())) / targets.size(0)
        
        # 3. Semantic Alignment Loss - L_sat
        if self.enable_sat:
            if batch.get('text_anch') is None:
                raise ValueError("Semantic alignment is enabled but text anchors are unavailable")
            rationale_norm = F.normalize(output['rationale'], dim=-1)
            text_anch_norm = F.normalize(batch['text_anch'], dim=-1)
            l_sat = 1.0 - F.cosine_similarity(rationale_norm, text_anch_norm).mean()
        else:
            l_sat = output['logits'].new_zeros(())
        
        # 4. Orthogonality Loss - L_ortho
        z_a = vib['z_aff'].flatten(1)
        z_e = vib['z_env'].flatten(1)
        sim = F.cosine_similarity(z_a, z_e, dim=-1)
        l_ortho = (sim ** 2).mean()

        # 5. Total Objective - L_total
        total = l_hce + \
                (self.weights['sat'] * l_sat) + \
                (self.weights['ortho'] * l_ortho) + \
                (self.weights['kl'] * l_kl)

        return total, l_hce, l_kl, l_sat, l_ortho
