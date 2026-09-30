"""Synthetic, CPU-only end-to-end smoke test for the VIBE data/model contract."""

from pathlib import Path
import sys
import tempfile

import h5py
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloading.vibe_dataset import VIBEDataset, vibe_collate_fn
from losses.loss_function import VIBE_Loss
from model.vibe_model import VIBE_Transformer


K, T, D = 8, 32, 768
GLOBAL_TOKENS = T * 49


def write_sample(path: Path, label: int, person_lengths, audio_length: int) -> None:
    """Write one canonical-schema synthetic sample with intentionally variable lengths."""
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("global_v_seq", data=torch.randn(1, GLOBAL_TOKENS, D).numpy())
        h5_file.create_dataset("env_feat", data=torch.randn(1, D).numpy())
        h5_file.create_dataset("text_anch", data=torch.randn(1, D).numpy())
        h5_file.create_dataset("audio_seq", data=torch.randn(audio_length, D).numpy())
        h5_file.create_dataset("physics_sync", data=torch.eye(len(person_lengths)).numpy())
        persons = h5_file.create_group("person_sequences")
        for index, length in enumerate(person_lengths):
            persons.create_dataset(f"p_{index}", data=torch.randn(length, D).numpy())
        h5_file.attrs["label"] = label


def main() -> None:
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix="vibe_smoke_") as temp_dir:
        root = Path(temp_dir)
        first = root / "sample_a_vibe.h5"
        second = root / "sample_b_vibe.h5"
        write_sample(first, label=0, person_lengths=[16, 32, 40], audio_length=20)
        write_sample(second, label=2, person_lengths=[12, 35], audio_length=41)

        dataset = VIBEDataset([first, second], max_k=K, max_t=T)
        batch = vibe_collate_fn([dataset[0], dataset[1]])
        assert batch is not None
        assert batch["local_p"].shape == (2, K, T, D)
        assert batch["global_v"].shape == (2, GLOBAL_TOKENS, D)
        assert batch["env_feat"].shape == (2, D)
        assert batch["audio_seq"].shape == (2, T, D)
        assert batch["gamma"].shape == (2, 1, 1)
        assert torch.isfinite(batch["gamma"]).all()

        model = VIBE_Transformer(input_dim=D, latent_dim=512, num_classes=3)
        criterion = VIBE_Loss(lambda_sat=0.5, lambda_ortho=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        output = model.forward_with_gamma(
            local_p=batch["local_p"],
            global_v=batch["global_v"],
            env_feat=batch["env_feat"],
            audio_seq=batch["audio_seq"],
            gamma=batch["gamma"],
        )
        assert output["logits"].shape == (2, 3)
        total, l_hce, l_kl, l_sat, l_ortho = criterion(output, batch["labels"], batch)
        for value in (total, l_hce, l_kl, l_sat, l_ortho):
            assert value.ndim == 0 and torch.isfinite(value), value

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

    print(
        "PASS: import, HDF5 loading, collation, K/T padding, gamma, forward, "
        "all loss terms, backward, and one AdamW step completed."
    )


if __name__ == "__main__":
    main()
