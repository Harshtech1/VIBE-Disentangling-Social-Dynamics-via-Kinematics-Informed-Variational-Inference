#!/usr/bin/env python3
"""CPU-only regression/integration tests for approved ablation switches.

This test never loads or writes HDF5 data or checkpoints.  It verifies that
default switch values preserve the current model path and that each proposed
intervention retains the reconstruction tensor contract on deterministic
synthetic inputs.
"""

from __future__ import annotations

import copy

import torch

from losses.loss_function import VIBE_Loss
from model.vibe_model import VIBE_Transformer


torch.set_num_threads(1)
torch.manual_seed(20260928)

MODEL_KWARGS = dict(
    input_dim=768,
    latent_dim=512,
    num_classes=3,
    attention_heads=8,
    transformer_depth=2,
    dropout=0.3,
    drop_path=0.1,
)


def inputs(batch_size=2):
    return {
        "local_p": torch.randn(batch_size, 8, 32, 768),
        "global_v": torch.randn(batch_size, 768),
        "audio_seq": torch.randn(batch_size, 32, 768),
        "gamma": torch.full((batch_size, 1, 1), 0.37),
        "agent_mask": torch.tensor([[True] * 8, [True] * 5 + [False] * 3]),
        "local_time_mask": torch.cat(
            [torch.ones(batch_size, 5, 32, dtype=torch.bool), torch.zeros(batch_size, 3, 32, dtype=torch.bool)], dim=1
        ),
        "audio_time_mask": torch.ones(batch_size, 32, dtype=torch.bool),
    }


def run(model, values):
    model.eval()
    with torch.no_grad():
        return model.forward_with_gamma(
            values["local_p"], values["global_v"], None, values["audio_seq"], values["gamma"],
            agent_mask=values["agent_mask"], local_time_mask=values["local_time_mask"],
            audio_time_mask=values["audio_time_mask"], reconstruction_mode=True,
        )


def assert_valid(output):
    assert output["logits"].shape == (2, 3)
    assert output["h_final"].shape == (2, 512)
    assert output["vib_out"]["z_aff"].shape == (2, 8, 32, 512)
    assert output["vib_out"]["z_env"].shape == (2, 8, 32, 512)
    for value in (output["logits"], output["h_final"], output["vib_out"]["z_aff"], output["vib_out"]["z_env"]):
        assert torch.isfinite(value).all()


def test_default_regression(values, baseline):
    explicit = VIBE_Transformer(
        **MODEL_KWARGS,
        enable_context_injection=True,
        gamma_conditioning="stored",
        enable_cross_modal_attention=True,
        audio_input_mode="observed",
        global_scene_conditioning="observed",
    )
    explicit.load_state_dict(copy.deepcopy(baseline.state_dict()))
    expected, actual = run(baseline, values), run(explicit, values)
    assert torch.equal(expected["logits"], actual["logits"])
    assert torch.equal(expected["h_final"], actual["h_final"])
    assert torch.equal(expected["vib_out"]["z_aff"], actual["vib_out"]["z_aff"])


def test_loss_switches():
    logits = torch.randn(2, 3)
    vib = {
        "mu_aff": torch.randn(2, 8, 32, 512),
        "logvar_aff": torch.randn(2, 8, 32, 512).clamp(-2, 2),
        "z_aff": torch.randn(2, 8, 32, 512),
        "z_env": torch.randn(2, 8, 32, 512),
    }
    output = {"logits": logits, "vib_out": vib, "rationale": torch.randn(2, 768), "mixup_data": None}
    targets = torch.tensor([0, 2])
    batch = {"text_anch": None}
    baseline = VIBE_Loss(lambda_sat=0.0, lambda_ortho=0.1, lambda_kl=1.0, enable_sat=False)
    total, hce, kl, sat, ortho = baseline(output, targets, batch)
    legacy_total = hce + (0.0 * sat) + (0.1 * ortho) + kl
    assert torch.allclose(total, legacy_total, rtol=0.0, atol=1e-6)
    no_ortho = VIBE_Loss(lambda_sat=0.0, lambda_ortho=0.0, lambda_kl=1.0, enable_sat=False)
    no_ortho_total, hce_o, kl_o, sat_o, ortho_o = no_ortho(output, targets, batch)
    assert torch.allclose(no_ortho_total, hce_o + kl_o + 0.0 * sat_o + 0.0 * ortho_o, rtol=0.0, atol=1e-6)
    no_kl = VIBE_Loss(lambda_sat=0.0, lambda_ortho=0.1, lambda_kl=0.0, enable_sat=False)
    no_kl_total, hce_k, kl_k, sat_k, ortho_k = no_kl(output, targets, batch)
    assert torch.allclose(no_kl_total, hce_k + 0.1 * ortho_k + 0.0 * sat_k, rtol=0.0, atol=1e-6)
    assert kl_k.item() == kl.item()  # Computation remains unchanged; only its weight changes.


def test_switches(values, baseline):
    initial_masks = {name: values[name].clone() for name in ("agent_mask", "local_time_mask", "audio_time_mask")}

    context = VIBE_Transformer(**MODEL_KWARGS, enable_context_injection=False)
    context.load_state_dict(copy.deepcopy(baseline.state_dict()))
    context_calls = []
    handle = context.context_gate.context_net.register_forward_hook(lambda *_: context_calls.append(True))
    assert_valid(run(context, values)); handle.remove()
    assert not context_calls

    gamma = VIBE_Transformer(**MODEL_KWARGS, gamma_conditioning="fixed_zero")
    gamma.load_state_dict(copy.deepcopy(baseline.state_dict()))
    seen_gamma = []
    handle = gamma.temporal_encoder.register_forward_pre_hook(lambda _m, args: seen_gamma.append(args[1].detach().clone()))
    assert_valid(run(gamma, values)); handle.remove()
    assert len(seen_gamma) == 1 and torch.count_nonzero(seen_gamma[0]) == 0

    cross = VIBE_Transformer(**MODEL_KWARGS, enable_cross_modal_attention=False)
    cross.load_state_dict(copy.deepcopy(baseline.state_dict()))
    cross_calls, self_calls = [], []
    h1 = cross.cross_modal_block.cross_attn.register_forward_hook(lambda *_: cross_calls.append(True))
    h2 = cross.cross_modal_block.self_attn.register_forward_hook(lambda *_: self_calls.append(True))
    assert_valid(run(cross, values)); h1.remove(); h2.remove()
    assert not cross_calls and self_calls

    audio = VIBE_Transformer(**MODEL_KWARGS, audio_input_mode="zero_projected_tokens")
    audio.load_state_dict(copy.deepcopy(baseline.state_dict()))
    seen_audio = []
    handle = audio.cross_modal_block.register_forward_pre_hook(lambda _m, args: seen_audio.append(args[1].detach().clone()))
    assert_valid(run(audio, values)); handle.remove()
    assert len(seen_audio) == 1 and torch.count_nonzero(seen_audio[0]) == 0

    global_scene = VIBE_Transformer(**MODEL_KWARGS, global_scene_conditioning="fixed_neutral")
    global_scene.load_state_dict(copy.deepcopy(baseline.state_dict()))
    seen_style, seen_scale = [], []
    def capture_neutral_scene(_module, args):
        seen_scale.append(args[1].detach().clone())
        seen_style.append(args[2].detach().clone())
        # Returning None is essential: a pre-hook return would replace inputs.
        return None

    handle = global_scene.temporal_encoder.register_forward_pre_hook(capture_neutral_scene)
    assert_valid(run(global_scene, values)); handle.remove()
    assert len(seen_style) == 1 and torch.count_nonzero(seen_style[0]) == 0
    # The model's existing [B,1] scene-scale × [B,1,1] gamma broadcast is
    # preserved. With neutral unit scale, the value delivered to AdaLN equals
    # the original gamma even under that legacy broadcasting convention.
    assert len(seen_scale) == 1 and torch.all(seen_scale[0] == values["gamma"][0, 0, 0])

    for name, before in initial_masks.items():
        assert torch.equal(values[name], before), f"{name} was mutated"


def main():
    values = inputs()
    baseline = VIBE_Transformer(**MODEL_KWARGS)
    test_default_regression(values, baseline)
    test_loss_switches()
    test_switches(values, baseline)
    print("BASELINE REGRESSION = PASS")
    print("C ORTHOGONALITY = PASS")
    print("D KL SWITCH = PASS")
    print("E CONTEXT SWITCH = PASS")
    print("F GAMMA SWITCH = PASS")
    print("G1 CROSS-ATTENTION SWITCH = PASS")
    print("G2 AUDIO SWITCH = PASS")
    print("H GLOBAL SCENE SWITCH = PASS")
    print("B NO-VIB = DESIGN DECISION REQUIRED")


if __name__ == "__main__":
    main()
