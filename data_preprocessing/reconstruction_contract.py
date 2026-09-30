"""Deterministic temporal and kinematic adapters for VIBE-GECV-RECON-v1.0.

These helpers implement explicitly approved reconstruction deviations.  They
do not encode claims about the original authors' unpublished preprocessing.
"""

from __future__ import annotations

import numpy as np


T_RECON = 32
D_RECON = 768
K_RECON = 8


def select_k8_duration_confidence_id(
    trajectories: dict[int, list[tuple[int, np.ndarray]]], confidence: dict[int, list[float]], k: int = K_RECON,
) -> dict[str, object]:
    """Approved v1.0 reconstruction selection; not an author policy."""
    observed = sorted(trajectories)
    mean_confidence = {
        pid: float(np.mean(confidence.get(pid, []))) if confidence.get(pid) else float("-inf")
        for pid in observed
    }
    ranked = sorted(observed, key=lambda pid: (-len(trajectories[pid]), -mean_confidence[pid], pid))
    return {
        "status": "CUSTOM_RECONSTRUCTION_DEVIATION",
        "rule": "descending observed duration, then descending mean detection confidence, then ascending numeric track ID",
        "observed_track_ids": observed,
        "observed_track_lengths": {str(pid): len(trajectories[pid]) for pid in observed},
        "mean_detection_confidence": {str(pid): mean_confidence[pid] for pid in observed},
        "selected_track_ids": ranked[:k],
        "selected_count": min(len(ranked), k),
    }


def shared_time_grid(decoded_frames: int, source_fps: float, t: int = T_RECON) -> dict[str, np.ndarray]:
    """Return an endpoint-inclusive source-time grid and nearest source frames."""
    if decoded_frames < 1 or source_fps <= 0:
        raise ValueError("decoded_frames and source_fps must be positive")
    duration = (decoded_frames - 1) / source_fps
    timestamps = np.linspace(0.0, duration, t, dtype=np.float64)
    source_frames = np.rint(timestamps * source_fps).astype(np.int64)
    source_frames = np.clip(source_frames, 0, decoded_frames - 1)
    return {
        "timestamps": timestamps,
        "source_frames": source_frames,
        "valid": np.ones(t, dtype=bool),
        "duration_seconds": np.asarray(duration, dtype=np.float64),
    }


def bin_pool(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool observations into midpoint-defined target-time bins.

    Every source observation belongs to exactly one bin.  Empty bins remain
    zero with validity False; no interpolation or first-T truncation occurs.
    """
    values = np.asarray(values, dtype=np.float32)
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != source_times.shape[0]:
        raise ValueError("values must be [N,D] and source_times must be [N]")
    if target_times.ndim != 1 or len(target_times) < 1:
        raise ValueError("target_times must be a non-empty vector")
    pooled = np.zeros((len(target_times), values.shape[1]), dtype=np.float32)
    valid = np.zeros(len(target_times), dtype=bool)
    if len(values) == 0:
        return pooled, valid
    midpoints = (target_times[:-1] + target_times[1:]) / 2.0
    bins = np.searchsorted(midpoints, source_times, side="right")
    for index in range(len(target_times)):
        members = values[bins == index]
        if len(members):
            pooled[index] = members.mean(axis=0, dtype=np.float32)
            valid[index] = True
    return pooled, valid


def map_local_tracks(
    per_track_features: dict[int, np.ndarray],
    per_track_times: dict[int, np.ndarray],
    selected_ids: list[int],
    target_times: np.ndarray,
    k: int = K_RECON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map selected variable-length tracks to [K,T,768] plus explicit masks."""
    features = np.zeros((k, len(target_times), D_RECON), dtype=np.float32)
    time_mask = np.zeros((k, len(target_times)), dtype=bool)
    agent_mask = np.zeros(k, dtype=bool)
    for slot, pid in enumerate(selected_ids[:k]):
        pooled, valid = bin_pool(per_track_features[pid], per_track_times[pid], target_times)
        if pooled.shape[1] != D_RECON:
            raise ValueError(f"Expected D={D_RECON}, got {pooled.shape[1]}")
        features[slot] = pooled
        time_mask[slot] = valid
        agent_mask[slot] = bool(valid.any())
    return features, time_mask, agent_mask


def shared_interval_synchrony(
    trajectories: dict[int, list[tuple[int, np.ndarray]]],
    selected_ids: list[int],
    frame_shape: tuple[int, int],
    k: int = K_RECON,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Synchrony on common sampled-frame intervals, a v1.0 reconstruction rule."""
    height, width = frame_shape
    interval_velocity: dict[int, dict[tuple[int, int], np.ndarray]] = {}
    for pid in selected_ids[:k]:
        points = {int(frame): np.asarray(box, dtype=np.float32) for frame, box in trajectories[pid]}
        velocities: dict[tuple[int, int], np.ndarray] = {}
        for frame in sorted(points):
            nxt = frame + 1
            if nxt not in points:
                continue
            first, second = points[frame], points[nxt]
            p0 = np.asarray([(first[0] + first[2] / 2) / width, (first[1] + first[3] / 2) / height])
            p1 = np.asarray([(second[0] + second[2] / 2) / width, (second[1] + second[3] / 2) / height])
            delta = p1 - p0
            velocities[(frame, nxt)] = delta / (np.linalg.norm(delta) + 1e-6)
        interval_velocity[pid] = velocities
    matrix = np.zeros((k, k), dtype=np.float32)
    pair_valid = np.zeros((k, k), dtype=bool)
    for i, left in enumerate(selected_ids[:k]):
        for j, right in enumerate(selected_ids[:k]):
            if j <= i:
                continue
            common = sorted(set(interval_velocity[left]) & set(interval_velocity[right]))
            if common:
                cosine = [float(np.dot(interval_velocity[left][key], interval_velocity[right][key])) for key in common]
                matrix[i, j] = matrix[j, i] = float(np.mean(cosine))
                pair_valid[i, j] = pair_valid[j, i] = True
    valid_values = matrix[pair_valid]
    gamma = float(valid_values.mean()) if len(valid_values) else 0.0
    return matrix, pair_valid, gamma
