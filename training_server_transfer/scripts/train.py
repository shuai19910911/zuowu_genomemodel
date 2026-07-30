#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint as checkpoint
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, IterableDataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing training dependency: "
        f"{exc.name}. Install the training environment first, for example "
        "`mamba install -n zuowu_genomemodel numpy pytorch pytorch-cuda -c pytorch -c nvidia`, "
        "then rerun this script."
    ) from exc


DNA_COMPLEMENT = torch.tensor([3, 2, 1, 0, 4, 5, 6], dtype=torch.long)
LOGIT_COMPLEMENT = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)
DEFAULT_REGION_LABELS = ["background", "coding", "gene_body", "promoter", "splice", "tes", "utr"]
REGION_IGNORE_INDEX = -100


class StopController:
    def __init__(self):
        self.requested = False
        self.signal_number = None

    def request(self, signal_number, _frame):
        self.requested = True
        self.signal_number = signal_number

    def install(self):
        signal.signal(signal.SIGTERM, self.request)
        signal.signal(signal.SIGINT, self.request)


def reverse_complement_tokens(input_ids):
    complement = DNA_COMPLEMENT.to(input_ids.device)
    return complement[input_ids.flip(1)]


def reverse_complement_logits(logits):
    complement = LOGIT_COMPLEMENT.to(logits.device)
    return logits.flip(1).index_select(dim=-1, index=complement)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def safe_relative_path(value, field_name):
    raw = str(value)
    path = Path(raw)
    raw_parts = raw.split("/")
    if raw == "" or path.is_absolute() or any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError(f"unsafe {field_name} path: {value!r}")
    return path


def resolve_under(root, relative_path, field_name):
    root = Path(root).resolve()
    candidate = (root / safe_relative_path(relative_path, field_name)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{field_name} escapes root: {relative_path!r}")
    return candidate


def safe_torch_load_checkpoint(path):
    try:
        checkpoint_payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch checkpoint loading requires weights_only=True support") from exc
    if not isinstance(checkpoint_payload, dict):
        raise RuntimeError(f"checkpoint is not a dict: {path}")
    for key in ("model", "step"):
        if key not in checkpoint_payload:
            raise RuntimeError(f"checkpoint missing required key {key!r}: {path}")
    return checkpoint_payload


def build_region_label_map(labels=None):
    labels = list(labels or DEFAULT_REGION_LABELS)
    return {str(label).strip().lower(): idx for idx, label in enumerate(labels)}


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return False, 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank, rank, world_size


def cleanup_distributed(enabled):
    if enabled:
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def seed_training_step(base_seed, step, rank):
    words = np.random.SeedSequence([
        int(base_seed), int(step), int(rank), 0x43524F50,
    ]).generate_state(4, dtype=np.uint32)
    python_seed = (int(words[0]) << 32) | int(words[1])
    numpy_seed = int(words[2])
    torch_seed = ((int(words[2]) << 32) | int(words[3])) & ((1 << 63) - 1)
    random.seed(python_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    return torch_seed


def read_manifest(stage_dir):
    rows = []
    with open(stage_dir / "manifest.tsv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            row["tokens"] = int(row["tokens"])
            row["windows_count"] = int(row["windows_count"])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No shards found in {stage_dir / 'manifest.tsv'}")
    return rows


def load_window_index(path, split, region_label_map=None):
    offsets, lengths, region_labels = [], [], []
    region_label_map = region_label_map or {}
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row["split"] != split:
                continue
            offsets.append(int(row["offset"]))
            lengths.append(int(row["length"]))
            bucket = str(row.get("region_bucket", "")).strip().lower()
            region_labels.append(int(region_label_map.get(bucket, REGION_IGNORE_INDEX)))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(lengths, dtype=np.int32),
        np.asarray(region_labels, dtype=np.int64),
    )


def deterministic_window_order(shard_lengths, seed):
    order = [(shard_idx, window_idx) for shard_idx, length in enumerate(shard_lengths) for window_idx in range(length)]
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    return order


class NoReplacementBatchPlan:
    """Deterministic global batches with no repeated window in one epoch.

    Complete batches contain one sequence length so every DDP rank receives
    equal-shaped work. Per-length remainders are shuffled into a small mixed
    tail so no training window is dropped.
    """

    _TAIL_BUCKET = -1

    def __init__(self, references_by_length, global_batch_size, seed):
        self.global_batch_size = int(global_batch_size)
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if not references_by_length:
            raise ValueError("references_by_length must not be empty")
        rng = np.random.default_rng(int(seed))
        self._bucket_lengths = sorted(int(length) for length in references_by_length)
        self._batches_by_length = {}
        descriptor_buckets = []
        descriptor_indices = []
        tail_references = []
        tail_lengths = []
        for bucket_idx, length in enumerate(self._bucket_lengths):
            references = np.asarray(references_by_length[length], dtype=np.uint64).reshape(-1)
            if references.size == 0:
                continue
            references = references[rng.permutation(references.size)]
            complete_count = references.size // self.global_batch_size
            complete_size = complete_count * self.global_batch_size
            complete = references[:complete_size].reshape(complete_count, self.global_batch_size)
            self._batches_by_length[length] = complete
            if complete_count:
                descriptor_buckets.append(np.full(complete_count, bucket_idx, dtype=np.int16))
                descriptor_indices.append(np.arange(complete_count, dtype=np.int64))
            remainder = references[complete_size:]
            if remainder.size:
                tail_references.append(remainder)
                tail_lengths.append(np.full(remainder.size, length, dtype=np.int32))

        self._tail_batches = []
        self._tail_batch_lengths = []
        if tail_references:
            tail_refs = np.concatenate(tail_references)
            tail_lens = np.concatenate(tail_lengths)
            order = rng.permutation(tail_refs.size)
            tail_refs = tail_refs[order]
            tail_lens = tail_lens[order]
            for start in range(0, tail_refs.size, self.global_batch_size):
                self._tail_batches.append(tail_refs[start:start + self.global_batch_size])
                self._tail_batch_lengths.append(tail_lens[start:start + self.global_batch_size])
            descriptor_buckets.append(np.full(len(self._tail_batches), self._TAIL_BUCKET, dtype=np.int16))
            descriptor_indices.append(np.arange(len(self._tail_batches), dtype=np.int64))

        if not descriptor_buckets:
            raise ValueError("references_by_length contains no windows")
        buckets = np.concatenate(descriptor_buckets)
        indices = np.concatenate(descriptor_indices)
        descriptor_order = rng.permutation(buckets.size)
        self._descriptor_buckets = buckets[descriptor_order]
        self._descriptor_indices = indices[descriptor_order]

    @property
    def total_batches(self):
        return int(self._descriptor_buckets.size)

    def _descriptor(self, position):
        position = int(position)
        if position < 0 or position >= self.total_batches:
            raise IndexError(f"batch position out of range: {position}")
        return int(self._descriptor_buckets[position]), int(self._descriptor_indices[position])

    def global_batch(self, position):
        bucket_idx, batch_idx = self._descriptor(position)
        if bucket_idx == self._TAIL_BUCKET:
            return self._tail_batches[batch_idx]
        length = self._bucket_lengths[bucket_idx]
        return self._batches_by_length[length][batch_idx]

    def batch_length(self, position):
        bucket_idx, batch_idx = self._descriptor(position)
        if bucket_idx == self._TAIL_BUCKET:
            lengths = self._tail_batch_lengths[batch_idx]
            return int(lengths[0]) if lengths.size and bool(np.all(lengths == lengths[0])) else None
        return self._bucket_lengths[bucket_idx]

    def local_batch(self, position, rank, world_size):
        rank = int(rank)
        world_size = int(world_size)
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
        chunks = np.array_split(self.global_batch(position), world_size)
        if any(chunk.size == 0 for chunk in chunks):
            raise RuntimeError("global tail batch is too small to give every DDP rank work")
        return chunks[rank]


def reference_catalog_sha256(references_by_length):
    digest = hashlib.sha256()
    digest.update(b"cropgenome-no-replacement-catalog-v1\0")
    for length in sorted(references_by_length):
        references = np.asarray(references_by_length[length], dtype="<u8").reshape(-1)
        digest.update(int(length).to_bytes(8, "little", signed=False))
        digest.update(int(references.size).to_bytes(8, "little", signed=False))
        digest.update(references.tobytes(order="C"))
    return digest.hexdigest()


class NoReplacementBatchStream:
    STATE_SCHEMA_VERSION = 1

    def __init__(self, dataset, micro_batch_size, rank, world_size, seed, state=None):
        self.dataset = dataset
        self.micro_batch_size = int(micro_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.world_size <= 0 or self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"invalid rank/world_size: {self.rank}/{self.world_size}")
        self.references_by_length = dataset.reference_catalog_by_length()
        self.catalog_sha256 = reference_catalog_sha256(self.references_by_length)
        self.global_batch_size = self.micro_batch_size * self.world_size
        self.epoch = 0
        self.batch_position = 0
        if state is not None:
            self._restore_state(state)
        self._build_plan()

    def _build_plan(self):
        self.plan = NoReplacementBatchPlan(
            self.references_by_length,
            global_batch_size=self.global_batch_size,
            seed=self.seed + self.epoch * 1000003,
        )
        if self.batch_position < 0 or self.batch_position > self.plan.total_batches:
            raise ValueError(
                f"sampler batch_position {self.batch_position} is outside epoch with "
                f"{self.plan.total_batches} batches"
            )

    def _restore_state(self, state):
        if not isinstance(state, dict):
            raise ValueError("sampler state must be a dictionary")
        required = {
            "schema_version", "seed", "global_batch_size", "catalog_sha256",
            "epoch", "batch_position",
        }
        if set(state) != required:
            raise ValueError(f"sampler state fields mismatch: {sorted(state)}")
        expected = {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "seed": self.seed,
            "global_batch_size": self.global_batch_size,
            "catalog_sha256": self.catalog_sha256,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"sampler state {key} mismatch: {state.get(key)!r} != {value!r}")
        self.epoch = int(state["epoch"])
        self.batch_position = int(state["batch_position"])
        if self.epoch < 0:
            raise ValueError("sampler epoch must be non-negative")

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_position >= self.plan.total_batches:
            self.epoch += 1
            self.batch_position = 0
            self._build_plan()
        position = self.batch_position
        references = self.plan.local_batch(position, self.rank, self.world_size)
        rng_seed = np.random.SeedSequence([
            self.seed, self.epoch, position, self.rank,
        ]).generate_state(1, dtype=np.uint64)[0]
        rng = np.random.default_rng(rng_seed)
        batch = [self.dataset.materialize_reference(reference, rng) for reference in references]
        self.batch_position += 1
        return batch

    def state_dict(self):
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "seed": self.seed,
            "global_batch_size": self.global_batch_size,
            "catalog_sha256": self.catalog_sha256,
            "epoch": self.epoch,
            "batch_position": self.batch_position,
        }


def iter_collated_batches(stream, collate_fn):
    while True:
        yield collate_fn(next(stream))


class StageWindowDataset(IterableDataset):
    def __init__(self, root, stage_dir, split, seed=1, rc_prob=0.5, region_label_map=None, deterministic=False):
        super().__init__()
        self.stage_dir = resolve_under(root, stage_dir, "stage_dir")
        self.split = split
        self.seed = seed
        self.rc_prob = rc_prob
        self.region_label_map = region_label_map or build_region_label_map()
        self.deterministic = deterministic
        self.shards = read_manifest(self.stage_dir)
        weights = np.asarray([max(1, row["windows_count"]) for row in self.shards], dtype=np.float64)
        self.shard_probs = weights / weights.sum()
        self._cache = {}

    def _get_shard(self, shard_idx):
        if shard_idx in self._cache:
            return self._cache[shard_idx]
        row = self.shards[shard_idx]
        input_path = resolve_under(self.stage_dir, row["input_ids"], "input_ids")
        windows_path = resolve_under(self.stage_dir, row["windows"], "windows")
        ids = np.memmap(input_path, dtype=np.uint8, mode="r")
        offsets, lengths, region_labels = load_window_index(windows_path, self.split, self.region_label_map)
        if len(offsets) == 0:
            offsets = np.asarray([0], dtype=np.int64)
            lengths = np.asarray([0], dtype=np.int32)
            region_labels = np.asarray([REGION_IGNORE_INDEX], dtype=np.int64)
        self._cache[shard_idx] = (ids, offsets, lengths, region_labels)
        return self._cache[shard_idx]

    def reference_catalog_by_length(self):
        buckets = {}
        for shard_idx in range(len(self.shards)):
            _, _, lengths, _ = self._get_shard(shard_idx)
            valid = lengths > 0
            if not bool(valid.any()):
                continue
            window_indices = np.arange(lengths.size, dtype=np.uint64)
            references = (np.uint64(shard_idx) << np.uint64(32)) | window_indices
            for length in np.unique(lengths[valid]):
                length = int(length)
                buckets.setdefault(length, []).append(references[lengths == length])
        if not buckets:
            raise RuntimeError(f"No {self.split} windows found in {self.stage_dir}")
        return {
            length: np.concatenate(parts).astype(np.uint64, copy=False)
            for length, parts in sorted(buckets.items())
        }

    def _materialize_window(self, shard_idx, window_idx, rng):
        ids, offsets, lengths, region_labels = self._get_shard(shard_idx)
        start = int(offsets[window_idx])
        length = int(lengths[window_idx])
        arr = np.asarray(ids[start:start + length], dtype=np.int64)
        seq = torch.from_numpy(arr.copy()).long()
        if rng.random() < self.rc_prob:
            seq = DNA_COMPLEMENT[seq.flip(0)]
        return seq, int(region_labels[window_idx])

    def materialize_reference(self, reference, rng):
        reference = int(reference)
        shard_idx = reference >> 32
        window_idx = reference & 0xFFFFFFFF
        if shard_idx < 0 or shard_idx >= len(self.shards):
            raise IndexError(f"window reference shard out of range: {reference}")
        return self._materialize_window(shard_idx, window_idx, rng)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng_seed = self.seed + worker_id
        if not self.deterministic:
            rng_seed += 9973 * int(time.time()) % 1000003
        rng = np.random.default_rng(rng_seed)
        if self.deterministic:
            shard_lengths = []
            for shard_idx in range(len(self.shards)):
                _, offsets, lengths, _ = self._get_shard(shard_idx)
                shard_lengths.append(0 if len(offsets) == 1 and lengths[0] == 0 else len(offsets))
            order = deterministic_window_order(shard_lengths, rng_seed)
            if not order:
                raise RuntimeError(f"No {self.split} windows found in {self.stage_dir}")
            while True:
                for shard_idx, window_idx in order:
                    yield self._materialize_window(shard_idx, window_idx, rng)
        while True:
            shard_idx = int(rng.choice(len(self.shards), p=self.shard_probs))
            ids, offsets, lengths, region_labels = self._get_shard(shard_idx)
            if len(offsets) == 1 and lengths[0] == 0:
                continue
            j = int(rng.integers(0, len(offsets)))
            yield self._materialize_window(shard_idx, j, rng)


def collate_and_mask(batch, mask_prob, mask_id=5, pad_id=6, force_mask_per_sequence=True):
    sequences = []
    region_values = []
    for item in batch:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            seq, region_label = item
        else:
            seq, region_label = item, REGION_IGNORE_INDEX
        sequences.append(seq)
        region_values.append(int(region_label))
    max_len = max(x.numel() for x in sequences)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        input_ids[i, :seq.numel()] = seq
        attention_mask[i, :seq.numel()] = True
    region_labels = torch.tensor(region_values, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    valid = attention_mask & (input_ids < 4)
    masked = (torch.rand_like(input_ids.float()) < mask_prob) & valid
    if force_mask_per_sequence:
        for row_idx in range(masked.shape[0]):
            if bool(valid[row_idx].any().item()) and not bool(masked[row_idx].any().item()):
                valid_positions = valid[row_idx].nonzero(as_tuple=False).flatten()
                chosen = valid_positions[torch.randint(0, valid_positions.numel(), (1,)).item()]
                masked[row_idx, chosen] = True
    labels[masked] = input_ids[masked]
    replace_mask = (torch.rand_like(input_ids.float()) < 0.80) & masked
    replace_rand = (torch.rand_like(input_ids.float()) < 0.50) & masked & ~replace_mask
    input_ids[replace_mask] = mask_id
    input_ids[replace_rand] = torch.randint(0, 5, (int(replace_rand.sum()),), dtype=torch.long)
    return input_ids, labels, attention_mask, region_labels


def next_dry_run_batch(train_iter, required_sequence_length=None, max_batches=10000):
    for scanned in range(1, int(max_batches) + 1):
        batch = next(train_iter)
        if required_sequence_length is None:
            return batch, scanned
        sequence_lengths = batch[2].sum(dim=1)
        if bool((sequence_lengths == int(required_sequence_length)).all().item()):
            return batch, scanned
    raise RuntimeError(
        f"dry_run could not find sequence_length={required_sequence_length} "
        f"within {max_batches} batches"
    )


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, ratio=2.0, dropout=0.05):
        super().__init__()
        hidden = max(dim, int(dim * ratio))
        self.norm = RMSNorm(dim)
        self.up_gate = nn.Linear(dim, hidden * 2)
        self.down = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x, gate = self.up_gate(self.norm(x)).chunk(2, dim=-1)
        x = self.down(F.silu(gate) * x)
        return residual + self.dropout(x)


class HyenaLiteBlock(nn.Module):
    def __init__(self, dim, expand=2, kernel_size=127, dropout=0.05, mlp_ratio=2.0):
        super().__init__()
        hidden = dim * expand
        self.norm = RMSNorm(dim)
        self.in_proj = nn.Linear(dim, hidden * 2)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden)
        self.out_proj = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = SwiGLUFeedForward(dim, ratio=mlp_ratio, dropout=dropout)

    def forward(self, x, attention_mask=None):
        residual = x
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        x = self.out_proj(F.silu(gate) * u)
        return self.ffn(residual + self.dropout(x))


class LocalSelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, chunk_size=512, dropout=0.05, dilation=1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"d_model={dim} must be divisible by attention_heads={num_heads}")
        self.norm = RMSNorm(dim)
        self.chunk_size = int(chunk_size)
        self.dilation = max(1, int(dilation))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        if self.chunk_size <= 0:
            return x
        residual = x
        x = self.norm(x)
        batch, seq_len, dim = x.shape
        chunk = min(self.chunk_size, seq_len)
        max_dilation = max(1, math.ceil(seq_len / chunk))
        dilation = min(self.dilation, max_dilation)
        superblock = chunk * dilation
        pad_len = (-seq_len) % superblock
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, pad_len), value=False)
        padded_len = x.shape[1]
        n_superblocks = padded_len // superblock
        x = x.reshape(batch, n_superblocks, chunk, dilation, dim)
        x = x.permute(0, 1, 3, 2, 4).reshape(batch * n_superblocks * dilation, chunk, dim)
        key_padding_mask = None
        if attention_mask is not None:
            valid = attention_mask.reshape(batch, n_superblocks, chunk, dilation)
            valid = valid.permute(0, 1, 3, 2).reshape(batch * n_superblocks * dilation, chunk)
            key_padding_mask = ~valid
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad] = False
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x.reshape(batch, n_superblocks, dilation, chunk, dim)
        x = x.permute(0, 1, 3, 2, 4).reshape(batch, padded_len, dim)
        if pad_len:
            x = x[:, :seq_len, :]
        return residual + self.dropout(x)


def make_mamba2_block(dim, cfg):
    try:
        from mamba_ssm.modules.mamba2 import Mamba2
        return nn.Sequential(
            RMSNorm(dim),
            Mamba2(
                d_model=dim,
                d_state=int(cfg.get("mamba_d_state", 128)),
                d_conv=int(cfg.get("mamba_d_conv", 4)),
                expand=int(cfg.get("expand", 2)),
            ),
        )
    except Exception:
        return None


class ResidualMamba2(nn.Module):
    def __init__(self, block, dropout):
        super().__init__()
        self.block = block
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        return x + self.dropout(self.block(x))


class CropGenomeFM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = int(cfg["d_model"])
        self.embed = nn.Embedding(7, dim, padding_idx=6)
        self.layers = nn.ModuleList()
        backend = "hyena_lite"
        attention_every = int(cfg.get("attention_every", 0))
        attention_dilations = [max(1, int(value)) for value in cfg.get("attention_dilations", [])]
        attention_idx = 0
        added_local_attention = False
        for layer_idx in range(int(cfg["n_layers"])):
            block = None
            if "mamba2" in cfg.get("backend_priority", []):
                block = make_mamba2_block(dim, cfg)
            if block is not None:
                self.layers.append(ResidualMamba2(block, float(cfg.get("dropout", 0.05))))
                backend = "mamba2"
            else:
                self.layers.append(HyenaLiteBlock(
                    dim=dim,
                    expand=int(cfg.get("expand", 2)),
                    kernel_size=int(cfg.get("conv_kernel", 127)),
                    dropout=float(cfg.get("dropout", 0.05)),
                    mlp_ratio=float(cfg.get("mlp_ratio", 2.0)),
                ))
            if attention_every > 0 and (layer_idx + 1) % attention_every == 0:
                dilation = attention_dilations[min(attention_idx, len(attention_dilations) - 1)] if attention_dilations else 1
                self.layers.append(LocalSelfAttentionBlock(
                    dim=dim,
                    num_heads=int(cfg.get("attention_heads", 8)),
                    chunk_size=int(cfg.get("attention_chunk_size", 512)),
                    dropout=float(cfg.get("dropout", 0.05)),
                    dilation=dilation,
                ))
                attention_idx += 1
                added_local_attention = True
        if added_local_attention:
            backend = f"{backend}+local_attn"
        self.backend = backend
        self.gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
        self.rc_equivariant = bool(cfg.get("rc_equivariant", False))
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, 5, bias=False)
        self.region_labels = list(cfg.get("region_labels", DEFAULT_REGION_LABELS))
        self.num_region_labels = len(self.region_labels)
        self.region_head = None
        self.region_loss_enabled = float(cfg.get("region_classification_weight", 0.0)) > 0.0
        if self.region_loss_enabled and self.num_region_labels > 0:
            self.region_head = nn.Sequential(
                RMSNorm(dim),
                nn.Linear(dim, self.num_region_labels),
            )

    def encode_hidden(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                if attention_mask is None:
                    def custom_forward(hidden, module=layer):
                        return module(hidden, None)
                    x = checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
                else:
                    def custom_forward(hidden, mask, module=layer):
                        return module(hidden, mask)
                    x = checkpoint.checkpoint(custom_forward, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)
        return x

    def logits_from_hidden(self, hidden):
        return self.head(self.norm(hidden))

    @staticmethod
    def sequence_pool(hidden, attention_mask=None):
        if attention_mask is None:
            return hidden.mean(dim=1)
        weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hidden * weights).sum(dim=1) / denom

    def forward(self, input_ids, attention_mask=None, return_aux=False):
        direct_hidden = self.encode_hidden(input_ids, attention_mask)
        direct_logits = self.logits_from_hidden(direct_hidden)
        hidden = direct_hidden
        if not self.rc_equivariant:
            if return_aux:
                region_logits = None
                if self.region_head is not None:
                    region_logits = self.region_head(self.sequence_pool(hidden, attention_mask))
                return direct_logits, {
                    "hidden": hidden,
                    "region_logits": region_logits,
                    "direct_logits": direct_logits,
                    "rc_logits_aligned": None,
                }
            return direct_logits

        rc_input_ids = reverse_complement_tokens(input_ids)
        rc_attention_mask = attention_mask.flip(1) if attention_mask is not None else None
        rc_hidden = self.encode_hidden(rc_input_ids, rc_attention_mask)
        rc_logits = self.logits_from_hidden(rc_hidden)
        rc_logits_aligned = reverse_complement_logits(rc_logits)
        hidden = 0.5 * (direct_hidden + rc_hidden.flip(1))
        logits = 0.5 * (direct_logits + rc_logits_aligned)
        if return_aux:
            region_logits = None
            if self.region_head is not None:
                region_logits = self.region_head(self.sequence_pool(hidden, attention_mask))
            return logits, {
                "hidden": hidden,
                "region_logits": region_logits,
                "direct_logits": direct_logits,
                "rc_logits_aligned": rc_logits_aligned,
            }
        return logits


def cosine_lr(step, cfg):
    lr = float(cfg["learning_rate"])
    min_lr = float(cfg["min_learning_rate"])
    warmup = int(cfg["warmup_steps"])
    max_steps = int(cfg["max_steps"])
    if step < warmup:
        return lr * max(1, step) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _model_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def _strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_model_state_safely(model, model_state, allow_partial=False):
    target = model.module if hasattr(model, "module") else model
    target_state = target.state_dict()
    incoming = _strip_module_prefix(model_state)
    report = {
        "missing_keys": [],
        "unexpected_keys": [],
        "skipped_shape_mismatch": [],
        "loaded_keys": [],
    }
    if not allow_partial:
        result = target.load_state_dict(incoming, strict=True)
        report["missing_keys"] = list(result.missing_keys)
        report["unexpected_keys"] = list(result.unexpected_keys)
        report["loaded_keys"] = list(incoming.keys())
        return report

    compatible = {}
    for key, value in incoming.items():
        if key not in target_state:
            report["unexpected_keys"].append(key)
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            report["skipped_shape_mismatch"].append(key)
            continue
        compatible[key] = value
    result = target.load_state_dict(compatible, strict=False)
    report["missing_keys"] = list(result.missing_keys)
    report["unexpected_keys"].extend(list(result.unexpected_keys))
    report["loaded_keys"] = list(compatible.keys())
    return report


def training_checkpoint_extra(train_stream, extra=None):
    payload = dict(extra or {})
    if train_stream is not None:
        payload["sampler_state"] = train_stream.state_dict()
        payload["rng_policy"] = "stateless_step_rank_v1"
    return payload


def save_checkpoint(path, model, optimizer, step, cfg, model_cfg, rank, filename=None, extra=None):
    if not is_main(rank):
        return
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": _model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "train_config": cfg,
        "model_config": model_cfg,
    }
    if extra:
        payload.update(extra)
    target = path / (filename or f"step_{step:08d}.pt")
    temporary = path / f"{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def count_pretraining_components(batches, device):
    counts = torch.zeros(3, dtype=torch.float64)
    for _, labels, attention_mask, region_labels in batches:
        counts[0] += labels.ne(-100).sum()
        counts[1] += attention_mask.sum()
        counts[2] += region_labels.ne(REGION_IGNORE_INDEX).sum()
    return counts.to(device=device)


def gradient_accumulation_sync_context(model, ddp_enabled, microbatch_index, accumulation_steps):
    microbatch_index = int(microbatch_index)
    accumulation_steps = int(accumulation_steps)
    if accumulation_steps <= 0 or microbatch_index < 0 or microbatch_index >= accumulation_steps:
        raise ValueError(
            f"invalid microbatch_index/accumulation_steps: {microbatch_index}/{accumulation_steps}"
        )
    if ddp_enabled and microbatch_index < accumulation_steps - 1:
        return model.no_sync()
    return nullcontext()


def weighted_pretraining_objective(losses, counts, global_counts, model_cfg, world_size=1):
    weights = {
        "mlm": float(model_cfg.get("mlm_loss_weight", 1.0)),
        "rc": float(model_cfg.get("rc_consistency_weight", 0.0)),
        "region": float(model_cfg.get("region_classification_weight", 0.0)),
    }
    reference = next(iter(losses.values()))
    objective = reference.new_zeros(())
    for name, loss in losses.items():
        local_count = torch.as_tensor(counts[name], device=loss.device, dtype=loss.dtype)
        global_count = torch.as_tensor(global_counts[name], device=loss.device, dtype=loss.dtype).clamp_min(1.0)
        objective = objective + weights[name] * loss * local_count / global_count
    return objective * float(world_size)


def summarize_pretraining_metrics(metric_sums, global_counts, model_cfg):
    def normalized(name):
        denominator = max(float(global_counts[name]), 1.0)
        return float(metric_sums[name]) / denominator

    mlm_loss = normalized("mlm")
    rc_loss = normalized("rc")
    region_loss = normalized("region")
    region_count = max(float(global_counts["region"]), 1.0)
    region_acc = float(metric_sums["region_correct"]) / region_count
    loss = (
        float(model_cfg.get("mlm_loss_weight", 1.0)) * mlm_loss
        + float(model_cfg.get("rc_consistency_weight", 0.0)) * rc_loss
        + float(model_cfg.get("region_classification_weight", 0.0)) * region_loss
    )
    return {
        "loss": loss,
        "mlm_loss": mlm_loss,
        "rc_loss": rc_loss,
        "region_loss": region_loss,
        "region_acc": region_acc,
        "region_valid_count": float(global_counts["region"]),
        "selection_loss": mlm_loss + float(model_cfg.get("rc_selection_weight", 0.0)) * rc_loss,
    }


def pretraining_loss(
    model,
    input_ids,
    labels,
    attention_mask,
    rc_weight,
    region_labels=None,
    region_weight=0.0,
    mlm_weight=1.0,
    region_label_smoothing=0.0,
    rc_selection_weight=0.0,
    return_components=False,
):
    logits, aux = model(input_ids, attention_mask, return_aux=True)
    if bool(labels.ne(-100).any().item()):
        mlm_loss = F.cross_entropy(logits.view(-1, 5), labels.view(-1), ignore_index=-100)
    else:
        mlm_loss = logits.sum() * 0.0
    rc_loss = logits.new_zeros(())
    direct_logits = aux.get("direct_logits") if aux else None
    rc_logits_aligned = aux.get("rc_logits_aligned") if aux else None
    if rc_weight > 0.0 and direct_logits is not None and rc_logits_aligned is not None:
        valid = attention_mask.bool()
        if bool(valid.any().item()):
            direct_log_prob = F.log_softmax(direct_logits, dim=-1)
            rc_log_prob = F.log_softmax(rc_logits_aligned, dim=-1)
            direct_prob = direct_log_prob.exp()
            rc_prob = rc_log_prob.exp()
            direct_to_rc = F.kl_div(direct_log_prob, rc_prob, reduction="none").sum(dim=-1)
            rc_to_direct = F.kl_div(rc_log_prob, direct_prob, reduction="none").sum(dim=-1)
            rc_loss = 0.5 * (direct_to_rc[valid].mean() + rc_to_direct[valid].mean())
    region_loss = logits.new_zeros(())
    region_acc = logits.new_zeros(())
    region_valid_count = logits.new_zeros(())
    region_logits = aux.get("region_logits") if aux else None
    if region_weight > 0.0 and region_labels is not None and region_logits is not None:
        valid_region = region_labels.ne(REGION_IGNORE_INDEX)
        region_valid_count = valid_region.sum().to(logits.dtype)
        if bool(valid_region.any().item()):
            region_loss = F.cross_entropy(
                region_logits[valid_region],
                region_labels[valid_region],
                label_smoothing=float(region_label_smoothing),
            )
            pred = region_logits[valid_region].argmax(dim=-1)
            region_acc = pred.eq(region_labels[valid_region]).float().mean()
        else:
            region_loss = region_logits.sum() * 0.0
    selection_loss = mlm_loss + float(rc_selection_weight) * rc_loss
    total_loss = float(mlm_weight) * mlm_loss + float(rc_weight) * rc_loss + float(region_weight) * region_loss
    metrics = {
        "loss": total_loss.detach(),
        "mlm_loss": mlm_loss.detach(),
        "rc_loss": rc_loss.detach(),
        "region_loss": region_loss.detach(),
        "region_acc": region_acc.detach(),
        "region_valid_count": region_valid_count.detach(),
        "selection_loss": selection_loss.detach(),
    }
    if return_components:
        components = {"mlm": mlm_loss, "rc": rc_loss, "region": region_loss}
        counts = {
            "mlm": labels.ne(-100).sum(),
            "rc": attention_mask.sum(),
            "region": region_labels.ne(REGION_IGNORE_INDEX).sum() if region_labels is not None else labels.new_zeros(()),
        }
        return total_loss, metrics, components, counts
    return total_loss, metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    mask_prob,
    batches,
    amp_dtype,
    rc_weight,
    region_weight=0.0,
    mlm_weight=1.0,
    region_label_smoothing=0.0,
    rc_selection_weight=0.0,
    eval_seed=None,
):
    model.eval()
    metric_sums = torch.zeros(4, dtype=torch.float64, device=device)
    count_sums = torch.zeros(3, dtype=torch.float64, device=device)
    cpu_rng_state = torch.random.get_rng_state()
    if eval_seed is not None:
        torch.manual_seed(int(eval_seed))
    try:
        eval_iter = iter(loader)
        for _ in range(batches):
            input_ids, labels, attention_mask, region_labels = next(eval_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            region_labels = region_labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                _, metrics, _, counts = pretraining_loss(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    rc_weight,
                    region_labels=region_labels,
                    region_weight=region_weight,
                    mlm_weight=mlm_weight,
                    region_label_smoothing=region_label_smoothing,
                    rc_selection_weight=rc_selection_weight,
                    return_components=True,
                )
            batch_counts = torch.stack([counts["mlm"], counts["rc"], counts["region"]]).to(torch.float64)
            metric_sums += torch.stack([
                metrics["mlm_loss"] * batch_counts[0],
                metrics["rc_loss"] * batch_counts[1],
                metrics["region_loss"] * batch_counts[2],
                metrics["region_acc"] * batch_counts[2],
            ]).to(torch.float64)
            count_sums += batch_counts
    finally:
        if eval_seed is not None:
            torch.random.set_rng_state(cpu_rng_state)
    model.train()
    values = torch.cat([metric_sums, count_sums])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    summary_cfg = {
        "mlm_loss_weight": mlm_weight,
        "rc_consistency_weight": rc_weight,
        "region_classification_weight": region_weight,
        "rc_selection_weight": rc_selection_weight,
    }
    return summarize_pretraining_metrics(
        {"mlm": values[0], "rc": values[1], "region": values[2], "region_correct": values[3]},
        {"mlm": values[4], "rc": values[5], "region": values[6]},
        summary_cfg,
    )


def update_best_tracking(val_metrics, step, best_selection, best_step, stale_evals, min_delta):
    selection = float(val_metrics.get("selection_loss", val_metrics.get("mlm_loss", val_metrics["loss"])))
    improved = selection < (best_selection - float(min_delta))
    if improved:
        return selection, step, 0, selection, True
    return best_selection, best_step, stale_evals + 1, selection, False


def restore_best_tracking_from_resume(resume_checkpoint, early_state_path, start_step, reset_tracking=False):
    """Restore best-checkpoint bookkeeping after a resumed training launch.

    Without this, a resumed run starts with best_selection=inf and can overwrite
    checkpoint_best.pt at the next validation even if the new validation is
    worse than the pre-resume best checkpoint.
    """
    best_selection = float("inf")
    best_step = int(start_step)
    stale_evals = 0
    source = "fresh"
    if reset_tracking:
        return best_selection, best_step, stale_evals, source
    if resume_checkpoint:
        checkpoint_best = resume_checkpoint.get("best_selection_loss")
        checkpoint_best_step = resume_checkpoint.get("best_step")
        if checkpoint_best is not None and math.isfinite(float(checkpoint_best)):
            best_selection = float(checkpoint_best)
            best_step = int(checkpoint_best_step if checkpoint_best_step is not None else start_step)
            stale_evals = int(resume_checkpoint.get("stale_evals", 0))
            source = "checkpoint"
    if source == "fresh" and early_state_path.exists():
        state = load_json(early_state_path)
        state_step = int(state.get("step", -1))
        state_best = state.get("best_selection_loss")
        if state_step <= int(start_step) and state_best is not None and math.isfinite(float(state_best)):
            best_selection = float(state_best)
            best_step = int(state.get("best_step", start_step))
            stale_evals = int(state.get("stale_evals", 0))
            source = "early_stopping_state"
    return best_selection, best_step, stale_evals, source


def resolve_resume_policy(cfg, mode):
    if mode == "warmstart":
        policy = {"allow_partial": False, "resume_optimizer": False, "reset_step": True}
    elif mode == "exact":
        policy = {"allow_partial": False, "resume_optimizer": True, "reset_step": False}
    elif mode == "config":
        allow_partial = bool(cfg.get("allow_partial_resume", False))
        resume_optimizer = bool(cfg.get("resume_optimizer", not allow_partial))
        policy = {
            "allow_partial": allow_partial,
            "resume_optimizer": resume_optimizer,
            "reset_step": bool(cfg.get("reset_step_on_resume", allow_partial and not resume_optimizer)),
        }
    else:
        raise ValueError(f"Unknown resume mode: {mode}")
    if policy["allow_partial"] and policy["resume_optimizer"]:
        raise RuntimeError("allow_partial_resume=true is incompatible with resume_optimizer=true")
    return policy


def validate_resume_contract(root, resume_path, checkpoint, cfg, resume_mode):
    contract = cfg.get("resume_contract")
    if contract is None:
        return None
    required = {"checkpoint", "checkpoint_step", "resume_mode", "sampler_epoch_starts_fresh"}
    missing = sorted(required - set(contract))
    if missing:
        raise RuntimeError(f"resume_contract is missing fields: {missing}")
    expected_path = resolve_under(root, contract["checkpoint"], "resume_contract.checkpoint")
    if Path(resume_path).resolve() != expected_path:
        raise RuntimeError(f"resume_contract checkpoint mismatch: {resume_path} != {expected_path}")
    expected_step = int(contract["checkpoint_step"])
    observed_step = int(checkpoint["step"])
    if observed_step != expected_step:
        raise RuntimeError(
            f"resume_contract checkpoint_step mismatch: {observed_step} != {expected_step}"
        )
    if str(resume_mode) != str(contract["resume_mode"]):
        raise RuntimeError(
            f"resume_contract resume_mode mismatch: {resume_mode} != {contract['resume_mode']}"
        )
    return contract


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--config", required=True)
    p.add_argument("--model-config", default="configs/model_large.json")
    p.add_argument("--resume", default="")
    p.add_argument("--resume-mode", choices=("config", "warmstart", "exact"), default="config")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    stop_controller = StopController()
    stop_controller.install()

    ddp, local_rank, rank, world_size = setup_distributed()
    root = Path(args.data_root).resolve()
    cfg = load_json(resolve_under(root, args.config, "config"))
    model_cfg = load_json(resolve_under(root, args.model_config, "model_config"))
    base_seed = int(cfg.get("seed", 1))
    seed = base_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    expected_world_size = cfg.get("expected_world_size")
    if expected_world_size is not None and world_size != int(expected_world_size):
        raise RuntimeError(
            f"world_size={world_size} does not match expected_world_size={expected_world_size}"
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rc_augment_prob = float(model_cfg.get("rc_augmentation_prob", 0.0))
    region_label_map = build_region_label_map(model_cfg.get("region_labels", DEFAULT_REGION_LABELS))
    train_ds = StageWindowDataset(root, cfg["stage_dir"], "train", seed=seed, rc_prob=rc_augment_prob, region_label_map=region_label_map)
    val_ds = StageWindowDataset(root, cfg["stage_dir"], "val", seed=seed + 1000, rc_prob=0.0, region_label_map=region_label_map, deterministic=True)
    collate = lambda b: collate_and_mask(
        b,
        float(model_cfg.get("mlm_probability", 0.15)),
        force_mask_per_sequence=bool(model_cfg.get("force_mask_per_sequence", True)),
    )
    sampler_mode = str(cfg.get("sampler_mode", "with_replacement"))
    if sampler_mode not in {"with_replacement", "global_no_replacement"}:
        raise ValueError(f"unknown sampler_mode: {sampler_mode}")
    train_loader = None
    train_stream = None
    train_iter = None
    if sampler_mode == "with_replacement":
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["micro_batch_size"]),
            num_workers=int(cfg.get("num_workers", 2)),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
        )
        train_iter = iter(train_loader)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    model = CropGenomeFM(model_cfg).to(device)
    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            static_graph=bool(cfg.get("ddp_static_graph", False)),
            gradient_as_bucket_view=bool(cfg.get("ddp_gradient_as_bucket_view", False)),
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    start_step = 0
    reset_step = False
    resume_checkpoint = None
    if args.resume:
        resume_path = resolve_under(root, args.resume, "resume")
        ckpt = safe_torch_load_checkpoint(resume_path)
        validate_resume_contract(root, resume_path, ckpt, cfg, args.resume_mode)
        resume_checkpoint = ckpt
        model_state = ckpt["model"]
        resume_policy = resolve_resume_policy(cfg, args.resume_mode)
        allow_partial = resume_policy["allow_partial"]
        resume_report = load_model_state_safely(model, model_state, allow_partial=allow_partial)
        resume_optimizer = resume_policy["resume_optimizer"]
        reset_step = resume_policy["reset_step"]
        if resume_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = 0 if reset_step else int(ckpt["step"])
        if is_main(rank):
            print(json.dumps({
                "resume": args.resume,
                "resume_mode": args.resume_mode,
                "allow_partial_resume": allow_partial,
                "resume_optimizer": resume_optimizer,
                "reset_step_on_resume": reset_step,
                "start_step": start_step,
                "loaded_keys": len(resume_report["loaded_keys"]),
                "missing_keys": resume_report["missing_keys"][:20],
                "unexpected_keys": resume_report["unexpected_keys"][:20],
                "skipped_shape_mismatch": resume_report["skipped_shape_mismatch"][:20],
            }), flush=True)

    if sampler_mode == "global_no_replacement":
        sampler_state = None if resume_checkpoint is None else resume_checkpoint.get("sampler_state")
        sampler_reset = sampler_state is None and start_step > 0
        contract_allows_reset = bool(
            cfg.get("resume_contract", {}).get("sampler_epoch_starts_fresh", False)
        )
        if sampler_reset and not (
            bool(cfg.get("allow_sampler_reset_on_resume", False)) and contract_allows_reset
        ):
            raise RuntimeError(
                "resume checkpoint has no sampler_state; set allow_sampler_reset_on_resume=true "
                "only when intentionally starting a new no-replacement epoch"
            )
        train_stream = NoReplacementBatchStream(
            train_ds,
            micro_batch_size=int(cfg["micro_batch_size"]),
            rank=rank,
            world_size=world_size,
            seed=int(cfg.get("sampler_seed", base_seed)),
            state=sampler_state,
        )
        train_iter = iter_collated_batches(train_stream, collate)
        if is_main(rank):
            print(json.dumps({
                "sampler_mode": sampler_mode,
                "sampler_reset_from_legacy_checkpoint": sampler_reset,
                "sampler_state": train_stream.state_dict(),
                "length_bucket_counts": {
                    str(length): int(references.size)
                    for length, references in train_stream.references_by_length.items()
                },
            }), flush=True)

    amp_dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float16
    out_dir = resolve_under(root, cfg["output_dir"], "output_dir")
    grad_accum = int(cfg["grad_accum_steps"])
    max_steps = int(cfg["max_steps"])
    rc_weight = float(model_cfg.get("rc_consistency_weight", 0.0))
    region_weight = float(model_cfg.get("region_classification_weight", 0.0))
    mlm_weight = float(model_cfg.get("mlm_loss_weight", 1.0))
    region_label_smoothing = float(model_cfg.get("region_label_smoothing", 0.0))
    rc_selection_weight = float(model_cfg.get("rc_selection_weight", rc_weight))
    eval_seed = int(cfg.get("eval_seed", seed + 2000))
    if is_main(rank):
        model_for_log = model.module if hasattr(model, "module") else model
        trainable_params = sum(p.numel() for p in model_for_log.parameters() if p.requires_grad)
        print(
            f"stage={cfg['stage']} backend={model_for_log.backend} "
            f"layers={len(model_for_log.layers)} params={trainable_params} "
            f"rc_equivariant={model_for_log.rc_equivariant} rc_weight={rc_weight} "
            f"region_weight={region_weight} world_size={world_size}"
        )
    if args.dry_run:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        required_sequence_length = cfg.get("dry_run_sequence_length")
        dry_run_batch, batches_scanned = next_dry_run_batch(
            train_iter,
            required_sequence_length=required_sequence_length,
            max_batches=int(cfg.get("dry_run_max_batches", 10000)),
        )
        input_ids, labels, attention_mask, region_labels = dry_run_batch
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        region_labels = region_labels.to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            loss, metrics = pretraining_loss(
                model,
                input_ids,
                labels,
                attention_mask,
                rc_weight,
                region_labels=region_labels,
                region_weight=region_weight,
                mlm_weight=mlm_weight,
                region_label_smoothing=region_label_smoothing,
                rc_selection_weight=rc_selection_weight,
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"dry_run produced non-finite loss: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if is_main(rank):
            dry_run_payload = {
                "dry_run": True,
                "batch_shape": tuple(input_ids.shape),
                "required_sequence_length": required_sequence_length,
                "batches_scanned": batches_scanned,
                "loss": float(loss.detach().cpu()),
                "mlm_loss": float(metrics["mlm_loss"].detach().cpu()),
                "rc_loss": float(metrics["rc_loss"].detach().cpu()),
                "region_loss": float(metrics["region_loss"].detach().cpu()),
                "region_acc": float(metrics["region_acc"].detach().cpu()),
                "region_valid_count": float(metrics["region_valid_count"].detach().cpu()),
                "selection_loss": float(metrics["selection_loss"].detach().cpu()),
                "device": str(device),
            }
            if device.type == "cuda":
                dry_run_payload["cuda_max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 1)
                dry_run_payload["cuda_max_memory_reserved_mib"] = round(torch.cuda.max_memory_reserved(device) / 1024 / 1024, 1)
            print(json.dumps(dry_run_payload), flush=True)
        cleanup_distributed(ddp)
        return

    model.train()
    optimizer.zero_grad(set_to_none=True)
    early_enabled = bool(cfg.get("early_stopping_enabled", False))
    early_min_steps = int(cfg.get("early_stopping_min_steps", 0))
    early_patience = int(cfg.get("early_stopping_patience_evals", 0))
    early_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    best_selection, best_step, stale_evals, best_restore_source = restore_best_tracking_from_resume(
        resume_checkpoint,
        out_dir / "early_stopping_state.json",
        start_step,
        reset_tracking=reset_step,
    )
    if is_main(rank):
        print(json.dumps({
            "best_tracking_source": best_restore_source,
            "best_step": best_step,
            "best_selection_loss": best_selection if math.isfinite(best_selection) else None,
            "stale_evals": stale_evals,
        }), flush=True)
    last_step = start_step
    for step in range(start_step + 1, max_steps + 1):
        if bool(cfg.get("stateless_step_rng", False)):
            seed_training_step(base_seed, step, rank)
        lr = cosine_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        step_batches = [next(train_iter) for _ in range(grad_accum)]
        local_count_values = count_pretraining_components(step_batches, device)
        global_count_values = local_count_values.clone()
        if ddp:
            dist.all_reduce(global_count_values, op=dist.ReduceOp.SUM)
        global_counts = {
            "mlm": global_count_values[0],
            "rc": global_count_values[1],
            "region": global_count_values[2],
        }
        metric_sums = torch.zeros(4, dtype=torch.float64, device=device)
        all_finite = torch.ones((), dtype=torch.bool, device=device)
        for microbatch_index, (input_ids, labels, attention_mask, region_labels) in enumerate(step_batches):
            with gradient_accumulation_sync_context(
                model,
                ddp_enabled=ddp,
                microbatch_index=microbatch_index,
                accumulation_steps=grad_accum,
            ):
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                region_labels = region_labels.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    _, metrics, components, counts = pretraining_loss(
                        model,
                        input_ids,
                        labels,
                        attention_mask,
                        rc_weight,
                        region_labels=region_labels,
                        region_weight=region_weight,
                        mlm_weight=mlm_weight,
                        region_label_smoothing=region_label_smoothing,
                        rc_selection_weight=rc_selection_weight,
                        return_components=True,
                    )
                    weighted_loss = weighted_pretraining_objective(
                        components,
                        counts,
                        global_counts,
                        model_cfg,
                        world_size=world_size,
                    )
                all_finite &= torch.stack([value.detach().float() for value in components.values()]).isfinite().all()
                weighted_loss.backward()
            batch_counts = torch.stack([counts["mlm"], counts["rc"], counts["region"]]).to(torch.float64)
            metric_sums += torch.stack([
                metrics["mlm_loss"] * batch_counts[0],
                metrics["rc_loss"] * batch_counts[1],
                metrics["region_loss"] * batch_counts[2],
                metrics["region_acc"] * batch_counts[2],
            ]).to(torch.float64)
        finite_value = all_finite.to(torch.int64)
        if ddp:
            dist.all_reduce(finite_value, op=dist.ReduceOp.MIN)
        if not bool(finite_value.item()):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"step {step} produced non-finite pretraining loss")
        metric_values = torch.cat([metric_sums, local_count_values])
        if ddp:
            dist.all_reduce(metric_values, op=dist.ReduceOp.SUM)
        running = summarize_pretraining_metrics(
            {"mlm": metric_values[0], "rc": metric_values[1], "region": metric_values[2], "region_correct": metric_values[3]},
            {"mlm": metric_values[4], "rc": metric_values[5], "region": metric_values[6]},
            model_cfg,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad_norm).item()):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"step {step} produced non-finite gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last_step = step
        stop_value = torch.tensor([1 if stop_controller.requested else 0], dtype=torch.int64, device=device)
        if ddp:
            dist.all_reduce(stop_value, op=dist.ReduceOp.MAX)
        if bool(stop_value.item()):
            interrupted_name = f"interrupted_step_{step:08d}.pt"
            save_checkpoint(
                out_dir / "checkpoints",
                model,
                optimizer,
                step,
                cfg,
                model_cfg,
                rank,
                filename=interrupted_name,
                extra=training_checkpoint_extra(
                    train_stream,
                    {"interrupted": True, "signal": stop_controller.signal_number},
                ),
            )
            if is_main(rank):
                print(json.dumps({"graceful_stop": True, "step": step, "checkpoint": interrupted_name}), flush=True)
            cleanup_distributed(ddp)
            return
        if is_main(rank) and step % 10 == 0:
            payload = {"step": step, "lr": lr}
            payload.update(running)
            print(json.dumps(payload), flush=True)
        if step % int(cfg["eval_every"]) == 0:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                float(model_cfg.get("mlm_probability", 0.15)),
                int(cfg["eval_batches"]),
                amp_dtype,
                rc_weight,
                region_weight=region_weight,
                mlm_weight=mlm_weight,
                region_label_smoothing=region_label_smoothing,
                rc_selection_weight=rc_selection_weight,
                eval_seed=eval_seed,
            )
            stop_training = False
            if is_main(rank):
                best_selection, best_step, stale_evals, selection, improved = update_best_tracking(
                    val_metrics,
                    step,
                    best_selection,
                    best_step,
                    stale_evals,
                    early_min_delta,
                )
                if improved:
                    save_checkpoint(
                        out_dir / "checkpoints",
                        model,
                        optimizer,
                        step,
                        cfg,
                        model_cfg,
                        rank,
                        filename="checkpoint_best.pt",
                        extra=training_checkpoint_extra(
                            train_stream,
                            {"best_step": best_step, "best_selection_loss": best_selection, "validation_metrics": val_metrics},
                        ),
                    )
                payload = {"step": step, "best_step": best_step, "best_selection_loss": best_selection, "stale_evals": stale_evals}
                payload.update({f"val_{key}": value for key, value in val_metrics.items()})
                print(json.dumps(payload), flush=True)
                state = {
                    "step": step,
                    "best_step": best_step,
                    "best_selection_loss": best_selection,
                    "stale_evals": stale_evals,
                    "early_stopping_enabled": early_enabled,
                    "early_stopping_min_steps": early_min_steps,
                    "early_stopping_patience_evals": early_patience,
                    "early_stopping_min_delta": early_min_delta,
                    "latest_validation_metrics": val_metrics,
                }
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "early_stopping_state.json", "w", encoding="utf-8") as fh:
                    json.dump(state, fh, ensure_ascii=False, indent=2, allow_nan=False)
                stop_training = early_enabled and step >= early_min_steps and early_patience > 0 and stale_evals >= early_patience
                if stop_training:
                    print(json.dumps({"early_stop": True, "step": step, "best_step": best_step, "best_selection_loss": best_selection}), flush=True)
            if ddp:
                stop_tensor = torch.tensor([1 if stop_training else 0], dtype=torch.int64, device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
            if stop_training:
                break
        if step % int(cfg["save_every"]) == 0:
            save_checkpoint(
                out_dir / "checkpoints", model, optimizer, step, cfg, model_cfg, rank,
                extra=training_checkpoint_extra(train_stream),
            )
    save_checkpoint(
        out_dir / "checkpoints", model, optimizer, last_step, cfg, model_cfg, rank,
        extra=training_checkpoint_extra(train_stream),
    )
    cleanup_distributed(ddp)


if __name__ == "__main__":
    main()