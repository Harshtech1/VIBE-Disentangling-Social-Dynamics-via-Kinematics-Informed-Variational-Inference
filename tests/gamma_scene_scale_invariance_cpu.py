"""CPU regression gate for the v1.1 per-sample gamma/scene-scale operation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from model.vibe_model import VIBE_Transformer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/train_gecv_reconstruction_v1_1.yaml"
CHECKPOINT = ROOT / "checkpoints/gecv_reconstruction_v1_runB/model_best.pth"
ATOL = 1e-6
RTOL = 1e-6


def build_model() -> VIBE_Transformer:
    config = yaml.safe_load(CONFIG.read_text())
    model = VIBE_Transformer(
        input_dim=config["input_dim"], latent_dim=config["latent_dim"],
        num_classes=config["num_classes"], attention_heads=config["attention_heads"],
        transformer_depth=config["transformer_depth"], dropout=config["dropout"],
        drop_path=config["drop_path"],
    ).eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    return model


def make_sample(offset: float) -> dict[str, torch.Tensor]:
    return {
        "local_p": torch.randn(8, 32, 768) + offset,
        "global_v": torch.randn(768) + offset,
        "audio_seq": torch.randn(32, 768) + offset,
        "gamma": torch.tensor([[0.2 + offset]], dtype=torch.float32),
        "agent_mask": torch.tensor([True, True, True, True, False, False, False, False]),
        "local_time_mask": torch.tensor([[True] * 24 + [False] * 8] * 4 + [[False] * 32] * 4),
        "audio_time_mask": torch.tensor([True] * 28 + [False] * 4),
    }


def batch(*samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


def run(model: VIBE_Transformer, inputs: dict[str, torch.Tensor], captured: dict | None = None) -> dict:
    handle = None
    if captured is not None:
        def hook(_module, args):
            captured["temporal_shape"] = list(args[0].shape)
            captured["gamma_prime_shape"] = list(args[1].shape)
            captured["gamma_prime"] = args[1].detach().clone()
        handle = model.temporal_encoder.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            return model.forward_with_gamma(
                inputs["local_p"], inputs["global_v"], None, inputs["audio_seq"], inputs["gamma"],
                agent_mask=inputs["agent_mask"], local_time_mask=inputs["local_time_mask"],
                audio_time_mask=inputs["audio_time_mask"], reconstruction_mode=True,
            )
    finally:
        if handle is not None:
            handle.remove()


def assert_close(left: torch.Tensor, right: torch.Tensor, label: str) -> float:
    delta = float((left - right).abs().max())
    if not torch.allclose(left, right, atol=ATOL, rtol=RTOL):
        raise AssertionError(f"{label} differs: max_abs={delta}")
    return delta


def main() -> None:
    torch.manual_seed(20260928)
    model = build_model()
    x, y = make_sample(0.0), make_sample(0.1)
    masks_before = {name: value.clone() for name, value in batch(x, y).items() if name.endswith("mask")}

    xy = batch(x, y)
    captured: dict = {}
    output_xy = run(model, xy, captured)
    output_xx = run(model, batch(x, x))
    output_yx = run(model, batch(y, x))

    with torch.no_grad():
        style = model.global_style_proj(xy["global_v"])
        scene_scale = model.scene_scaler(style).unsqueeze(-1)
    expected_gamma_prime = xy["gamma"] * scene_scale
    if tuple(xy["gamma"].shape) != (2, 1, 1):
        raise AssertionError(f"gamma shape changed: {tuple(xy['gamma'].shape)}")
    if tuple(scene_scale.shape) != (2, 1, 1):
        raise AssertionError(f"scene_scale shape changed: {tuple(scene_scale.shape)}")
    if tuple(captured["gamma_prime_shape"]) != (2, 1, 1):
        raise AssertionError(f"gamma_prime shape changed: {captured['gamma_prime_shape']}")
    if tuple(captured["temporal_shape"]) != (2, 32, 512):
        raise AssertionError(f"temporal shape changed: {captured['temporal_shape']}")
    assert_close(captured["gamma_prime"], expected_gamma_prime, "gamma_prime mathematical operation")
    if tuple(output_xy["logits"].shape) != (2, 3):
        raise AssertionError(f"logit shape changed: {tuple(output_xy['logits'].shape)}")
    if tuple(model.classifier[0].weight.shape[1:]) != (513,):
        raise AssertionError("classifier fusion width changed")
    for name, tensor in {**output_xy, "scene_scale": scene_scale, "expected_gamma_prime": expected_gamma_prime}.items():
        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
            raise AssertionError(f"Non-finite tensor: {name}")
    for name, before in masks_before.items():
        if not torch.equal(xy[name], before):
            raise AssertionError(f"Mask was mutated: {name}")

    companion_delta = assert_close(output_xx["logits"][0], output_xy["logits"][0], "same-sample companion invariance")
    order_delta = assert_close(output_xy["logits"][0], output_yx["logits"][1], "batch order invariance")
    result = {
        "status": "PASS",
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "shapes": {"gamma": list(xy["gamma"].shape), "scene_scale": list(scene_scale.shape), "gamma_prime": captured["gamma_prime_shape"], "temporal": captured["temporal_shape"], "fusion_width": 513, "logits": list(output_xy["logits"].shape)},
        "batch_invariance": {"same_sample_companion_max_abs": companion_delta, "order_max_abs": order_delta},
        "finiteness": "PASS", "mask_immutability": "PASS", "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
