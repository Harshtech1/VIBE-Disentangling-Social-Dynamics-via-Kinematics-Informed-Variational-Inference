"""HDF5-to-tensor adapter for the VIBE-GECV reconstruction protocol.

Data flow:
    one reconstruction HDF5 record -> padded features/masks -> collated batch
    dictionary -> ``model/vibe_model.py``.

The fixed storage interface uses K=8 maximum agent slots, T=32 temporal bins,
and 768-dimensional pretrained feature vectors. Boolean masks, not zero values
alone, identify padding. ``selected_agent_ids`` is provenance metadata rather
than a model input because its length legitimately varies by sample.
"""

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class VIBEDataset(Dataset):
    """Load one VIBE sample while enforcing the selected HDF5 schema.

    Reconstruction model inputs include local [8,32,768], audio [32,768],
    global scene [768], masks, synchrony matrix [8,8], and stored gamma.
    Labels are canonical: Positive=0, Neutral=1, Negative=2.
    """
    def __init__(self, file_list, max_k=8, max_t=32, schema_mode='auto', reconstruction_gamma_source='stored', effective_k=None):
        """
        Modified to accept a direct list of file paths.
        This allows complex splitting logic (like Group K-Fold) to happen outside.
        """
        self.files = file_list
        self.max_k = max_k 
        self.max_t = max_t 
        if schema_mode not in {'auto', 'legacy', 'gecv_reconstruction_v1'}:
            raise ValueError(f"Unknown schema_mode: {schema_mode}")
        if reconstruction_gamma_source not in {'stored', 'derive_from_sync'}:
            raise ValueError(f"Unknown reconstruction_gamma_source: {reconstruction_gamma_source}")
        self.schema_mode = schema_mode
        self.reconstruction_gamma_source = reconstruction_gamma_source
        if effective_k is not None and not 1 <= int(effective_k) <= self.max_k:
            raise ValueError(f"effective_k must be in [1, {self.max_k}], got {effective_k}")
        self.effective_k = None if effective_k is None else int(effective_k)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Read one HDF5 file and return named tensors for a later collate step.

        The method validates schema-specific assumptions early. Returning
        ``None`` only occurs after a reported file-reading error; the collator
        then removes that item rather than constructing a malformed batch.
        """
        h5_path = self.files[idx]
        try:
            with h5py.File(h5_path, 'r') as f:
                is_reconstruction = (
                    f.attrs.get('protocol', '') == 'VIBE-GECV-RECON-v1.0'
                    or ('global_scene' in f and 'audio_features' in f)
                )
                if self.schema_mode == 'legacy' and is_reconstruction:
                    raise ValueError(f"Reconstruction HDF5 rejected by schema_mode=legacy: {h5_path}")
                if self.schema_mode == 'gecv_reconstruction_v1' and not is_reconstruction:
                    raise ValueError(f"Non-reconstruction HDF5 rejected by schema_mode=gecv_reconstruction_v1: {h5_path}")

                if is_reconstruction:
                    # VIBE-GECV-RECON-v1.0 compatibility adapter.  The extractor
                    # intentionally stores a single [768] scene-calibration vector.
                    # It is passed as global_v [768] and is NOT duplicated as env_feat.
                    global_v = torch.from_numpy(f['global_scene'][:])
                    env_feat = None
                    text_anch = None
                    if f.attrs.get('text_status', 'OFF') != 'OFF':
                        raise ValueError(f"Reconstruction text must be OFF: {h5_path}")
                else:
                    global_v = torch.from_numpy(f['global_v_seq'][:]).squeeze(0)
                    env_feat = torch.from_numpy(f['env_feat'][:]).squeeze(0)
                    text_anch = torch.from_numpy(f['text_anch'][:]).squeeze(0) if 'text_anch' in f else None

                if 'local_features' in f:
                    local_p_tensor = torch.from_numpy(f['local_features'][:])
                    local_time_mask = torch.from_numpy(f['local_time_mask'][:]).bool()
                    agent_mask = torch.from_numpy(f['agent_mask'][:]).bool()
                    synchrony_pair_mask = (
                        torch.from_numpy(f['synchrony_pair_mask'][:]).bool()
                        if 'synchrony_pair_mask' in f else None
                    )
                    if tuple(local_p_tensor.shape) != (self.max_k, self.max_t, 768):
                        raise ValueError(f"Invalid reconstruction local_features shape {tuple(local_p_tensor.shape)}")
                    real_k = int(agent_mask.sum().item())
                else:
                    person_seqs = []
                    p_group = f['person_sequences']
                    p_keys = sorted(p_group.keys(), key=lambda x: int(x.split('_')[1]))
                    for pid in p_keys[:self.max_k]:
                        p_data = torch.from_numpy(p_group[pid][:])
                        curr_t = p_data.shape[0]
                        if curr_t > self.max_t:
                            p_data = p_data[:self.max_t]
                        elif curr_t < self.max_t:
                            p_data = torch.cat([p_data, torch.zeros(self.max_t - curr_t, 768)], dim=0)
                        person_seqs.append(p_data)
                    real_k = len(person_seqs)
                    while len(person_seqs) < self.max_k:
                        person_seqs.append(torch.zeros(self.max_t, 768))
                    local_p_tensor = torch.stack(person_seqs)
                    agent_mask = torch.arange(self.max_k) < real_k
                    local_time_mask = local_p_tensor.abs().sum(dim=-1) != 0
                    synchrony_pair_mask = None

                sync_mat = torch.from_numpy(f['physics_sync'][:])
                padded_sync = torch.zeros(self.max_k, self.max_k)
                k_limit = min(sync_mat.shape[0], self.max_k)
                padded_sync[:k_limit, :k_limit] = sync_mat[:k_limit, :k_limit]
                if synchrony_pair_mask is None:
                    synchrony_pair_mask = torch.ones(self.max_k, self.max_k, dtype=torch.bool)
                    synchrony_pair_mask.fill_diagonal_(False)
                elif tuple(synchrony_pair_mask.shape) != (self.max_k, self.max_k):
                    raise ValueError(f"Invalid reconstruction synchrony_pair_mask shape {tuple(synchrony_pair_mask.shape)}")

                # Effective-K experiments retain the physical K=8 HDF5 schema.
                # Only the first deterministically stored slots may participate;
                # all later slots are inactive everywhere downstream via masks.
                if self.effective_k is not None:
                    slot_mask = torch.arange(self.max_k) < self.effective_k
                    agent_mask = agent_mask & slot_mask
                    local_time_mask = local_time_mask & agent_mask.unsqueeze(-1)
                real_k = int(agent_mask.sum().item())

                audio_key = 'audio_features' if is_reconstruction else 'audio_seq'
                audio_seq = torch.from_numpy(f[audio_key][:])
                if audio_seq.shape[0] > self.max_t:
                    audio_seq = audio_seq[:self.max_t]
                elif audio_seq.shape[0] < self.max_t:
                    pad_a = torch.zeros(self.max_t - audio_seq.shape[0], 768)
                    audio_seq = torch.cat([audio_seq, pad_a], dim=0)
                audio_time_mask = torch.from_numpy(f['audio_time_mask'][:]).bool() if 'audio_time_mask' in f else (audio_seq.abs().sum(dim=-1) != 0)
                global_time_mask = torch.from_numpy(f['global_time_mask'][:]).bool() if 'global_time_mask' in f else None
                if f.attrs.get('protocol', '') == 'VIBE-GECV-RECON-v1.0' and not audio_time_mask.any():
                    raise ValueError(f"Reconstruction sample has no valid audio bins: {h5_path}")
                if is_reconstruction:
                    if tuple(sync_mat.shape) != (self.max_k, self.max_k):
                        raise ValueError(f"Invalid reconstruction physics_sync shape {tuple(sync_mat.shape)}")
                    if tuple(audio_seq.shape) != (self.max_t, 768):
                        raise ValueError(f"Invalid reconstruction audio_features shape {tuple(audio_seq.shape)}")
                    if torch.any(~agent_mask & local_time_mask.any(dim=1)):
                        raise ValueError(f"Masked reconstruction agent has valid local time positions: {h5_path}")
                    if not torch.isfinite(global_v).all() or not torch.isfinite(audio_seq[audio_time_mask]).all():
                        raise ValueError(f"Non-finite reconstruction global/audio feature: {h5_path}")
                    if self.reconstruction_gamma_source == 'stored' and self.effective_k is None:
                        if 'gamma' not in f or np.asarray(f['gamma'][()]).shape != ():
                            raise ValueError(f"Reconstruction scalar gamma is missing or invalid: {h5_path}")
                        sample_gamma = torch.as_tensor(f['gamma'][()], dtype=torch.float32)
                        if not torch.isfinite(sample_gamma):
                            raise ValueError(f"Non-finite reconstruction gamma: {h5_path}")
                    else:
                        sample_gamma = None
                else:
                    sample_gamma = None

                # Canonical VIBE labels: 0=Positive, 1=Neutral, 2=Negative.
                label = int(f.attrs['label'])
                if label not in (0, 1, 2):
                    raise ValueError(
                        f"Invalid canonical label {label} in {h5_path}; expected 0, 1, or 2."
                    )

            return {
                'local_p': local_p_tensor.float(), 'global_v': global_v.float(), 
                'env_feat': None if env_feat is None else env_feat.float(), 'audio_seq': audio_seq.float(),
                'sync_mat': padded_sync.float(), 'text_anch': None if text_anch is None else text_anch.float(),
                'label': torch.tensor(label, dtype=torch.long),
                'real_k': torch.tensor(real_k, dtype=torch.long),
                'agent_mask': agent_mask, 'local_time_mask': local_time_mask,
                'synchrony_pair_mask': synchrony_pair_mask,
                'audio_time_mask': audio_time_mask, 'global_time_mask': global_time_mask,
                'sample_gamma': sample_gamma,
                'schema_mode': 'gecv_reconstruction_v1' if is_reconstruction else 'legacy',
            }
        except Exception as e:
            print(f"Error loading {h5_path}: {e}")
            return None

def compute_masked_gamma(sync_mat, agent_mask, synchrony_pair_mask=None):
    """Average only valid, non-self synchrony pairs between active agents.

    Inputs are ``sync_mat`` [B,K,K], ``agent_mask`` [B,K], and optional valid
    pair mask [B,K,K]. Output is [B,1,1]: exactly one contextual synchrony
    scalar per sample. Gamma summarizes motion-direction similarity; it is not
    an emotion label or a ground-truth physical measurement.
    """
    B, K, _ = sync_mat.shape
    active_pairs = agent_mask.bool().unsqueeze(2) & agent_mask.bool().unsqueeze(1)
    non_self = ~torch.eye(K, dtype=torch.bool, device=sync_mat.device).unsqueeze(0)
    valid_pairs = active_pairs & non_self
    if synchrony_pair_mask is not None:
        valid_pairs = valid_pairs & synchrony_pair_mask.bool()
    gammas = []
    for i in range(B):
        values = sync_mat[i][valid_pairs[i]]
        gammas.append(values.mean() if values.numel() else sync_mat.new_zeros(()))
    return torch.stack(gammas).view(B, 1, 1)

def vibe_collate_fn(batch):
    """Stack individual records into the batch dictionary consumed by training.

    Stored reconstruction gamma is preferred because it was computed during
    extraction from physical-time valid pairs. The derived path exists for
    compatible records without stored gamma. A mixed batch is rejected because
    silently mixing those semantics would harm reproducibility.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None

    global_v = torch.stack([b['global_v'] for b in batch])
    env_values = [b['env_feat'] for b in batch]
    if all(value is None for value in env_values):
        env_feat = None
    elif any(value is None for value in env_values):
        raise ValueError("A batch cannot mix samples with and without env_feat")
    else:
        env_feat = torch.stack(env_values)
    local_p = torch.stack([b['local_p'] for b in batch])
    sync_mat = torch.stack([b['sync_mat'] for b in batch])
    text_anch = None if all(b['text_anch'] is None for b in batch) else torch.stack([b['text_anch'] for b in batch])
    audio_seq = torch.stack([b['audio_seq'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    real_k = torch.stack([b['real_k'] for b in batch])
    agent_mask = torch.stack([b['agent_mask'] for b in batch])
    local_time_mask = torch.stack([b['local_time_mask'] for b in batch])
    synchrony_pair_mask = torch.stack([b['synchrony_pair_mask'] for b in batch])
    audio_time_mask = torch.stack([b['audio_time_mask'] for b in batch])
    global_time_mask = None if any(b['global_time_mask'] is None for b in batch) else torch.stack([b['global_time_mask'] for b in batch])

    stored_gamma = [b['sample_gamma'] for b in batch]
    if all(value is not None for value in stored_gamma):
        # This is the extractor's physical-time, valid-pair reconstruction gamma.
        # Do not recompute it from zero-filled synchrony entries in the loader.
        gamma = torch.stack(stored_gamma).view(len(batch), 1, 1)
        gamma_source = 'stored_reconstruction_gamma'
    elif all(value is None for value in stored_gamma):
        gamma = compute_masked_gamma(sync_mat, agent_mask, synchrony_pair_mask)
        gamma_source = 'derived_from_active_valid_sync_pairs'
    else:
        raise ValueError("A batch cannot mix stored reconstruction gamma with derived gamma")

    return {
        'global_v': global_v,
        'env_feat': env_feat,
        'local_p': local_p,
        'sync_mat': sync_mat,
        'gamma': gamma,
        'text_anch': text_anch,
        'audio_seq': audio_seq,
        'labels': labels,
        'real_k': real_k,
        'agent_mask': agent_mask,
        'local_time_mask': local_time_mask,
        'synchrony_pair_mask': synchrony_pair_mask,
        'audio_time_mask': audio_time_mask,
        'global_time_mask': global_time_mask,
        'gamma_source': gamma_source,
    }

import random
from collections import defaultdict
from pathlib import Path

def create_group_split(data_root_path, val_ratio=0.2):
    """
    Splits data such that all clips from the same video (Group) stay together.
    Prevents data leakage from training to validation.
    """
    train_source = Path(data_root_path) / 'train'
    all_files = sorted(list(train_source.glob("*_vibe.h5")))
    
    # 1. Group files by Video ID (Prefix before the first underscore)
    # Example: '1_1_vibe.h5' -> '1', '5_2_vibe.h5' -> '5'
    video_groups = defaultdict(list)
    for f in all_files:
        # Extract ID: "5_1_vibe.h5" -> "5"
        video_id = f.name.split('_')[0]
        video_groups[video_id].append(f)
    
    # 2. Shuffle Unique Video IDs
    unique_videos = list(video_groups.keys())
    random.shuffle(unique_videos)
    
    # 3. Split based on Video Count (not clip count)
    split_idx = int(len(unique_videos) * (1 - val_ratio))
    train_vids = unique_videos[:split_idx]
    val_vids = unique_videos[split_idx:]
    
    # 4. Flatten back to file lists
    train_files = [f for vid in train_vids for f in video_groups[vid]]
    val_files = [f for vid in val_vids for f in video_groups[vid]]
    
    print(f"Splitting Complete:")
    print(f"  > Unique Videos: {len(unique_videos)}")
    print(f"  > Training:   {len(train_vids)} videos -> {len(train_files)} clips")
    print(f"  > Validation: {len(val_vids)} videos -> {len(val_files)} clips")
    
    return train_files, val_files
