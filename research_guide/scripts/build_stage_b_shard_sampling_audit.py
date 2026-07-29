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
DEFAULT_REGION_OUTPUT = ROOT / "research_guide" / "source_data" / "stage_b_region_training_coverage.tsv"
DEFAULT_SPLIT_OUTPUT = ROOT / "research_guide" / "source_data" / "stage_b_split_assembly_coverage.tsv"
CONTEXTS = (4096, 8192, 16384)
SPLITS = ("train", "val", "test")
REGIONS = ("background", "coding", "gene_body", "promoter", "splice", "tes", "utr")
STEP_DRAWS = {14000: 503_000, 17000: 611_000}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit Stage B shard packing and the historical two-stage training sampler."
    )
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--region-output", default=str(DEFAULT_REGION_OUTPUT))
    parser.add_argument("--split-output", default=str(DEFAULT_SPLIT_OUTPUT))
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
    region_counts = Counter()
    train_region_counts = Counter()
    assembly_ids = {split: set() for split in SPLITS}
    species_labels = {split: set() for split in SPLITS}
    with gzip.open(str(windows_path), "rt", encoding="utf-8", newline="") as handle:
        for window in csv.DictReader(handle, delimiter="\t"):
            split = window["split"]
            context = int(window["context"])
            region = window["region_bucket"]
            if split not in SPLITS:
                raise RuntimeError("unexpected split {!r} in {}".format(split, windows_path))
            if context not in CONTEXTS:
                raise RuntimeError("unexpected context {!r} in {}".format(context, windows_path))
            if region not in REGIONS:
                raise RuntimeError("unexpected region {!r} in {}".format(region, windows_path))
            split_counts[split] += 1
            context_counts[context] += 1
            region_counts[region] += 1
            assembly_ids[split].add(window["assembly_id"])
            if window.get("species"):
                species_labels[split].add(window["species"])
            if split == "train":
                train_region_counts[region] += 1

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

    result = {
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
    for region in REGIONS:
        result["all_{}".format(region)] = region_counts[region]
        result["train_{}".format(region)] = train_region_counts[region]
    result["_assembly_ids"] = assembly_ids
    result["_species_labels"] = species_labels
    return result


def main():
    args = parse_args()
    stage_dir = Path(args.stage_dir).resolve()
    output = Path(args.output).resolve()
    region_output = Path(args.region_output).resolve()
    split_output = Path(args.split_output).resolve()
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
        "train_fraction", "relative_train_sampling_weight",
    ]
    fields.extend("all_{}".format(region) for region in REGIONS)
    fields.extend("train_{}".format(region) for region in REGIONS)
    fields.extend([
        "input_sha256", "windows_sha256",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in audited:
            formatted = {field: row[field] for field in fields}
            formatted["mean_bp_per_window"] = "{:.6f}".format(row["mean_bp_per_window"])
            formatted["train_fraction"] = "{:.9f}".format(row["train_fraction"])
            formatted["relative_train_sampling_weight"] = "{:.9f}".format(
                row["relative_train_sampling_weight"]
            )
            writer.writerow(formatted)
    temporary.replace(output)

    region_fields = [
        "region", "all_pool_windows", "train_pool_windows", "train_pool_fraction",
        "historical_sampler_probability_per_draw", "expected_draws_step14000",
        "expected_draws_step17000", "actual_draw_count_recorded", "evidence_boundary",
    ]
    region_output.parent.mkdir(parents=True, exist_ok=True)
    region_temporary = region_output.with_suffix(region_output.suffix + ".tmp")
    with region_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=region_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for region in REGIONS:
            all_region_windows = sum(row["all_{}".format(region)] for row in audited)
            train_region_windows = sum(row["train_{}".format(region)] for row in audited)
            sampler_probability = sum(
                row["windows_total"] / all_windows
                * row["train_{}".format(region)] / row["train_windows"]
                for row in audited
            )
            writer.writerow({
                "region": region,
                "all_pool_windows": all_region_windows,
                "train_pool_windows": train_region_windows,
                "train_pool_fraction": "{:.9f}".format(train_region_windows / train_windows),
                "historical_sampler_probability_per_draw": "{:.9f}".format(sampler_probability),
                "expected_draws_step14000": "{:.3f}".format(STEP_DRAWS[14000] * sampler_probability),
                "expected_draws_step17000": "{:.3f}".format(STEP_DRAWS[17000] * sampler_probability),
                "actual_draw_count_recorded": "no",
                "evidence_boundary": "expected_from_verified_sampler_not_an_observed_draw_log",
            })
    region_temporary.replace(region_output)

    assemblies_by_split = {
        split: set().union(*(row["_assembly_ids"][split] for row in audited))
        for split in SPLITS
    }
    species_by_split = {
        split: set().union(*(row["_species_labels"][split] for row in audited))
        for split in SPLITS
    }
    split_fields = [
        "split", "assemblies_with_windows", "species_with_windows", "windows",
        "assembly_overlap_with_other_splits",
    ]
    split_output.parent.mkdir(parents=True, exist_ok=True)
    split_temporary = split_output.with_suffix(split_output.suffix + ".tmp")
    with split_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=split_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for split in SPLITS:
            writer.writerow({
                "split": split,
                "assemblies_with_windows": len(assemblies_by_split[split]),
                "species_with_windows": len(species_by_split[split]),
                "windows": sum(row["{}_windows".format(split)] for row in audited),
                "assembly_overlap_with_other_splits": len(
                    assemblies_by_split[split]
                    & set().union(*(assemblies_by_split[other] for other in SPLITS if other != split))
                ),
            })
    split_temporary.replace(split_output)

    full_shards = [row for row in audited if row["tokens"] >= 900_000_000]
    payload = {
        "status": "passed",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "output": str(output),
        "region_output": str(region_output),
        "split_output": str(split_output),
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
        "regions": {
            region: {
                "all_pool_windows": sum(row["all_{}".format(region)] for row in audited),
                "train_pool_windows": sum(row["train_{}".format(region)] for row in audited),
                "historical_sampler_probability_per_draw": sum(
                    row["windows_total"] / all_windows
                    * row["train_{}".format(region)] / row["train_windows"]
                    for row in audited
                ),
            }
            for region in REGIONS
        },
        "split_assembly_coverage": {
            split: {
                "assemblies_with_windows": len(assemblies_by_split[split]),
                "species_with_windows": len(species_by_split[split]),
            }
            for split in SPLITS
        },
        "assembly_split_overlap": {
            "train_val": len(assemblies_by_split["train"] & assemblies_by_split["val"]),
            "train_test": len(assemblies_by_split["train"] & assemblies_by_split["test"]),
            "val_test": len(assemblies_by_split["val"] & assemblies_by_split["test"]),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
