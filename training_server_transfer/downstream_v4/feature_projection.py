"""Frozen parameter-free feature projection for equal-budget downstream heads."""

import hashlib

import numpy as np
import torch


HEAD_INPUT_DIM = 256
PROJECTION_SEED = 20260801
PROJECTION_PROTOCOL_ID = "signed_balanced_feature_hash_256_v1"


def projection_spec(source_dim, output_dim=HEAD_INPUT_DIM, seed=PROJECTION_SEED):
    source_dim = int(source_dim); output_dim = int(output_dim); seed = int(seed)
    if source_dim <= 0 or output_dim <= 0:
        raise ValueError("projection dimensions must be positive")
    offset = seed % output_dim
    # 205 is coprime to 256, so dimensions <=256 map without collisions.
    multiplier = 205 if output_dim == 256 else 1
    buckets = ((np.arange(source_dim, dtype=np.int64) % output_dim) * multiplier + offset) % output_dim
    signs = np.empty(source_dim, dtype=np.float32)
    for index in range(source_dim):
        digest = hashlib.sha256(f"{seed}|{index}".encode("ascii")).digest()
        signs[index] = 1.0 if (digest[0] & 1) else -1.0
    counts = np.bincount(buckets, minlength=output_dim).astype(np.float32)
    scales = signs / np.sqrt(np.maximum(1.0, counts[buckets]))
    return buckets, scales


def project_numpy(features, output_dim=HEAD_INPUT_DIM, seed=PROJECTION_SEED):
    values = np.asarray(features, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("features must have a final feature dimension")
    buckets, scales = projection_spec(values.shape[-1], output_dim, seed)
    projected = np.zeros((*values.shape[:-1], int(output_dim)), dtype=np.float32)
    for source_index, bucket in enumerate(buckets):
        projected[..., bucket] += values[..., source_index] * scales[source_index]
    return projected


def project_torch(features, output_dim=HEAD_INPUT_DIM, seed=PROJECTION_SEED):
    if features.ndim < 2:
        raise ValueError("features must have a final feature dimension")
    buckets, scales = projection_spec(features.shape[-1], output_dim, seed)
    bucket_tensor = torch.as_tensor(buckets, dtype=torch.long, device=features.device)
    scale_tensor = torch.as_tensor(scales, dtype=features.dtype, device=features.device)
    projected = torch.zeros(
        (*features.shape[:-1], int(output_dim)), dtype=features.dtype, device=features.device,
    )
    projected.index_add_(-1, bucket_tensor, features * scale_tensor)
    return projected
