"""Deterministic non-neural sequence baselines in the common embedding format."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import read_canonical_rows, sha256_path


RC = str.maketrans("ACGT", "TGCA")


def _canonical_kmer(kmer):
    reverse = kmer.translate(RC)[::-1]
    return min(kmer, reverse)


def sequence_features(sequence, k_values=(3, 4, 5, 6), hash_dim=2048):
    sequence = sequence.upper()
    values = np.zeros(hash_dim + 3, dtype=np.float32)
    valid_bases = sum(base in "ACGT" for base in sequence)
    values[hash_dim] = (sequence.count("G") + sequence.count("C")) / max(1, valid_bases)
    values[hash_dim + 1] = len(sequence) / 8192.0
    values[hash_dim + 2] = sequence.count("N") / max(1, len(sequence))
    total = 0
    for k in k_values:
        for start in range(0, len(sequence) - k + 1):
            kmer = sequence[start:start + k]
            if set(kmer) - set("ACGT"):
                continue
            canonical = _canonical_kmer(kmer)
            index = int.from_bytes(hashlib.sha256(f"{k}|{canonical}".encode()).digest()[:8], "big") % hash_dim
            values[index] += 1.0
            total += 1
    if total:
        values[:hash_dim] /= total
    return values


def _center_context(sequence, context):
    if context is None or len(sequence) <= int(context):
        return sequence
    start = (len(sequence) - int(context)) // 2
    return sequence[start:start + int(context)]


def build_kmer_cache(dataset_root, output_path, task_id, hash_dim=2048, context=None):
    dataset_root = Path(dataset_root).resolve(); output_path = Path(output_path).resolve()
    rows = read_canonical_rows(dataset_root / "samples.tsv")
    features = np.stack([
        sequence_features(_center_context(row["sequence"], context), hash_dim=hash_dim)
        for row in rows
    ]).astype(np.float32)
    labels = np.asarray([row["label"] for row in rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, embeddings=features,
        sample_ids=np.asarray([row["sample_id"] for row in rows]), labels=labels,
        splits=np.asarray([row["split"] for row in rows]),
        species=np.asarray([row["species"] for row in rows]),
        assemblies=np.asarray([row.get("assembly_id") or row["species"] for row in rows]),
        group_ids=np.asarray([
            row.get("group_id") or row.get("assembly_id") or row["species"]
            for row in rows
        ]),
    )
    receipt = {
        "status": "ok", "task_id": task_id, "baseline": "KmerHash_GC",
        "hash_dim": int(hash_dim), "k_values": [3, 4, 5, 6], "rows": len(rows),
        "context_bp": int(context) if context is not None else None,
        "samples_sha256": sha256_path(dataset_root / "samples.tsv"),
        "cache": str(output_path), "cache_sha256": sha256_path(output_path),
    }
    (output_path.parent / "baseline_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt
