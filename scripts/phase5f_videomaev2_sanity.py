#!/usr/bin/env python3
"""One synthetic 16-frame VideoMAEv2 sanity check; no dataset media."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor


REPO_ID = "OpenGVLab/VideoMAEv2-Base"


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run only from the researcher-owned H200 terminal.")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    config = AutoConfig.from_pretrained(REPO_ID, trust_remote_code=True)
    processor = VideoMAEImageProcessor.from_pretrained(REPO_ID)
    model = AutoModel.from_pretrained(REPO_ID, config=config, trust_remote_code=True).to(device).eval()

    # Synthetic RGB frames intentionally test loading and input/output shape only.
    frames = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(16)]
    inputs = processor(frames, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 1, 3, 4).to(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs)
    torch.cuda.synchronize(device)
    tensor = output if isinstance(output, torch.Tensor) else (output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0])
    print(json.dumps({
        "status": "PASS",
        "model_id": REPO_ID,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": list(inputs["pixel_values"].shape),
        "output_type": type(output).__name__,
        "output_shape": list(tensor.shape),
        "elapsed_inference_seconds": round(time.perf_counter() - started, 4),
        "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
