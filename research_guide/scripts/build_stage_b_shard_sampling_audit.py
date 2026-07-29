#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_DIR = ROOT / "training_server_transfer" / "inputs" / "Stage_B"
DEFAULT_OUTPUT = ROOT / "research_guide" / "source_data" / "stage_b_shard_sampling_audit.tsv"
CONTEXTS = (4096, 8192, 16384)
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit Stage B shard packing and the historical two-stage training sampler."
    )
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(stage_dir):
    manifest_path = stage_dir / "manifest.tsv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError("empty Stage B manifest: {}".format(manifest_path))
    return manifest_path, rows


def audit_shard(stage_dir, row):
    windows_path = stage_dir / row["windows"]
    observed_hash = sha256_file(windows_path)
    if observed_hash != row["windows_sha256"]:
        raise RuntimeError("window index SHA-256 mismatch: {}".format(windows_path))

    split_counts = Counter()
    context_counts = Counter()
    with gzip.open(str(windows_path), "rt", encoding="utf-8", newline="") as handle:
        for window in csv.DictReader(handle, delimiter="\t"):
            split = window["split"]
            context = int(window["context"])
            if split not in SPLITS:
                raise RuntimeError("unexpected split {!r} in {}".format(split, windows_path))
            if context not in CONTEXTS:
                raise RuntimeError("unexpected context {!r} in {}".format(context, windows_path))
            split_counts[split] += 1
            context_counts[context] += 1

    windows_total = int(row["windows_count"])
    tokens = int(row["tokens"])
    observed_windows = sum(split_counts.values())
    observed_tokens = sum(context * context_counts[context] for context in CONTEXTS)
    if observed_windows != windows_total:
        raise RuntimeError(
            "window count mismatch for {}: manifest={} observed={}".format(
                row["shard"], windows_total, observed_windows
            )
        )
    if observed_tokens != tokens:
        raise RuntimeError(
            "token count mismatch for {}: manifest={} observed={}".format(
                row["shard"], tokens, observed_tokens
            )
        )
    if split_counts["train"] <= 0:
        raise RuntimeError("shard has no train windows: {}".format(row["shard"]))

    return {
        "shard": row["shard"],
        "tokens": tokens,
        "windows_total": windows_total,
        "train_windows": split_counts["train"],
        "val_windows": split_counts["val"],
        "test_windows": split_counts["test"],
        "context_4096": context_counts[4096],
        "context_8192": context_counts[8192],
        "context_16384": context_counts[16384],
        "mean_bp_per_window": tokens / windows_total,
        "train_fraction": split_counts["train"] / windows_total,
        "input_sha256": row["input_sha256"],
        "windows_sha256": row["windows_sha256"],
    }


def main():
    args = parse_args()
    stage_dir = Path(args.stage_dir).resolve()
    output = Path(args.output).resolve()
    manifest_path, manifest_rows = load_manifest(stage_dir)
    audited = [audit_shard(stage_dir, row) for row in manifest_rows]

    all_windows = sum(row["windows_total"] for row in audited)
    train_windows = sum(row["train_windows"] for row in audited)
    for row in audited:
        row["relative_train_sampling_weight"] = (
            row["windows_total"] * train_windows
            / (all_windows * row["train_windows"])
        )

    fields = [
        "shard", "tokens", "windows_total", "train_windows", "val_windows", "test_windows",
        "context_4096", "context_8192", "context_16384", "mean_bp_per_window",
        "train_fraction", "relative_train_sampling_weight", "input_sha256", "windows_sha256",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in audited:
            formatted = dict(row)
            formatted["mean_bp_per_window"] = "{:.6f}".format(row["mean_bp_per_window"])
            formatted["train_fraction"] = "{:.9f}".format(row["train_fraction"])
            formatted["relative_train_sampling_weight"] = "{:.9f}".format(
                row["relative_train_sampling_weight"]
            )
            writer.writerow(formatted)
    temporary.replace(output)

    full_shards = [row for row in audited if row["tokens"] >= 900_000_000]
    payload = {
        "status": "passed",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "output": str(output),
        "shards": len(audited),
        "tokens_total": sum(row["tokens"] for row in audited),
        "windows_total": all_windows,
        "train_windows": train_windows,
        "val_windows": sum(row["val_windows"] for row in audited),
        "test_windows": sum(row["test_windows"] for row in audited),
        "full_shard_token_range": [
            min(row["tokens"] for row in full_shards),
            max(row["tokens"] for row in full_shards),
        ],
        "full_shard_window_range": [
            min(row["windows_total"] for row in full_shards),
            max(row["windows_total"] for row in full_shards),
        ],
        "relative_train_sampling_weight_range": [
            min(row["relative_train_sampling_weight"] for row in audited),
            max(row["relative_train_sampling_weight"] for row in audited),
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
