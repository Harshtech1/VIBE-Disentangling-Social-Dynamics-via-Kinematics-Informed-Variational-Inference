"""Narrow, version-pinned backbone compatibility helpers for reconstruction."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from transformers import HubertModel


@contextmanager
def _legacy_hubert_weight_norm():
    """Construct HuBERT with checkpoint-compatible ``weight_g``/``weight_v`` keys.

    Transformers 4.38 selects PyTorch's newer parametrization-based
    ``weight_norm`` when running under Torch 2.8.  The public
    ``facebook/hubert-base-ls960`` checkpoint instead stores the legacy key
    pair.  The temporary construction-time override makes the model layout
    match those checkpoint keys; no global site-package file is edited.
    """
    parametrizations = torch.nn.utils.parametrizations
    modern_weight_norm = parametrizations.weight_norm
    delattr(parametrizations, "weight_norm")
    try:
        yield
    finally:
        setattr(parametrizations, "weight_norm", modern_weight_norm)


def load_hubert_base(model_id: str, device: str | torch.device, *, local_files_only: bool = True):
    """Load HuBERT with every pretrained checkpoint key accounted for."""
    with _legacy_hubert_weight_norm():
        model, info = HubertModel.from_pretrained(
            model_id, output_loading_info=True, local_files_only=local_files_only,
        )
    missing = info.get("missing_keys", [])
    unexpected = info.get("unexpected_keys", [])
    mismatched = info.get("mismatched_keys", [])
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "HuBERT checkpoint did not load exactly: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    return model.to(device).eval(), {
        "checkpoint_loading": "PASS",
        "weight_norm_compatibility": "project-local legacy weight_norm construction for Torch 2.8 + Transformers 4.38",
    }
