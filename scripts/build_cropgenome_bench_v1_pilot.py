#!/usr/bin/env python3
"""Build a small CropGenome-Bench v1 pilot dataset from Stage_B windows.

This is a pipeline dry-run, not the final benchmark dataset. Labels are derived
from existing Stage_B region_bucket annotations so the downstream evaluation can
exercise fixed splits, sequence loading, baselines, model embeddings, and output
contracts before the full GFF-derived hard-negative benchmark is frozen.
"""

import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

TASKS = {
    "promoter_TSS": {
        "title": "Promoter/TSS pilot proxy",
        "positive_buckets": {"promoter"},
        "negative_buckets": {"background"},
        "positive_definition": "Stage_B promoter windows as TSS/promoter proxy positives",
        "negative_definition": "Stage_B background windows as intergenic proxy negatives",
    },
    "TES_polyA": {
        "title": "TES/polyA pilot proxy",
        "positive_buckets": {"tes"},
        "negative_buckets": {"background"},
        "positive_definition": "Stage_B TES windows as transcription termination/polyA proxy positives",
        "negative_definition": "Stage_B background windows as intergenic proxy negatives",
    },
    "splice_acceptor": {
        "title": "Splice-site pilot proxy",
        "positive_buckets": {"splice"},
        "negative_buckets": {"gene_body", "coding"},
        "positive_definition": "Stage_B splice-flank windows as splice-site proxy positives",
        "negative_definition": "Stage_B gene-body/coding windows as non-splice genic proxy negatives",
    },
}


def safe_rel(value: str, field: str) -> Path:
    raw = str(value)
    path = Path(raw)
    if raw == "" or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe {field}: {value!r}")
    return path


def resolve_under(root: Path, rel: str, field: str) -> Path:
    root = root.resolve()
    path = (root / safe_rel(rel, field)).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{field} escapes root: {rel!r}")
    return path


def read_manifest(stage_dir: Path):
    with (stage_dir / "manifest.tsv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            yield row


def add_reservoir(buckets, seen, key, item, limit, rng):
    seen[key] += 1
    if len(buckets[key]) < limit:
        buckets[key].append(item)
        return
    idx = rng.randrange(seen[key])
    if idx < limit:
        buckets[key][idx] = item


def build_samples(stage_dir: Path, max_train_per_label: int, max_eval_per_label: int, context: int, seed: int):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    seen = Counter()
    manifest_rows = list(read_manifest(stage_dir))
    for manifest_row in manifest_rows:
        input_path = resolve_under(stage_dir, manifest_row["input_ids"], "input_ids")
        windows_path = resolve_under(stage_dir, manifest_row["windows"], "windows")
        with gzip.open(windows_path, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                split = row.get("split", "")
                if split not in {"train", "val"}:
                    continue
                try:
                    row_context = int(row.get("context", "0"))
                    length = int(row.get("length", "0"))
                except ValueError:
                    continue
                if row_context != context or length <= 0:
                    continue
                bucket = row.get("region_bucket", "").strip().lower()
                for task_id, spec in TASKS.items():
                    if bucket in spec["positive_buckets"]:
                        label = 1
                        label_name = "positive"
                    elif bucket in spec["negative_buckets"]:
                        label = 0
                        label_name = "negative"
                    else:
                        continue
                    limit = max_train_per_label if split == "train" else max_eval_per_label
                    key = (task_id, split, label)
                    item = {
                        "task_id": task_id,
                        "split": split,
                        "label": str(label),
                        "label_name": label_name,
                        "input_path": str(input_path),
                        "offset": row["offset"],
                        "length": row["length"],
                        "stage": row.get("stage", "Stage_B"),
                        "context": row.get("context", ""),
                        "region_bucket": bucket,
                        "source_region_type": row.get("source_region_type", ""),
                        "assembly_id": row.get("assembly_id", ""),
                        "species": row.get("species", ""),
                        "genus": row.get("genus", ""),
                        "contig_id": row.get("contig_id", ""),
                        "start0": row.get("start0", ""),
                        "end0": row.get("end0", ""),
                    }
                    add_reservoir(buckets, seen, key, item, limit, rng)
    samples_by_task = defaultdict(list)
    for task_id in TASKS:
        for split in ["train", "val"]:
            for label in [0, 1]:
                rows = buckets.get((task_id, split, label), [])
                rng.shuffle(rows)
                for idx, item in enumerate(rows):
                    item = dict(item)
                    item["sample_id"] = f"{task_id}.{split}.{label}.{idx:06d}"
                    samples_by_task[task_id].append(item)
    return samples_by_task, seen


def write_tsv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", default="training_server_transfer/inputs/Stage_B")
    parser.add_argument("--out-dir", default="training_server_transfer/runs/cropgenome_bench_v1_pilot/datasets")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-train-per-label", type=int, default=1024)
    parser.add_argument("--max-eval-per-label", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    samples_by_task, seen = build_samples(stage_dir, args.max_train_per_label, args.max_eval_per_label, args.context, args.seed)
    fieldnames = [
        "sample_id", "task_id", "split", "label", "label_name", "input_path", "offset", "length", "stage", "context",
        "region_bucket", "source_region_type", "assembly_id", "species", "genus", "contig_id", "start0", "end0",
    ]
    summary_rows = []
    for task_id, rows in samples_by_task.items():
        task_dir = out_dir / task_id
        write_tsv(task_dir / "samples.tsv", rows, fieldnames)
        counts = Counter((r["split"], r["label"]) for r in rows)
        manifest = {
            "status": "ok",
            "benchmark_id": "CropGenome-Bench-v1",
            "mode": "pilot_proxy_from_stage_b_region_buckets",
            "task_id": task_id,
            "task_title": TASKS[task_id]["title"],
            "positive_definition": TASKS[task_id]["positive_definition"],
            "negative_definition": TASKS[task_id]["negative_definition"],
            "context": args.context,
            "seed": args.seed,
            "max_train_per_label": args.max_train_per_label,
            "max_eval_per_label": args.max_eval_per_label,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
            "samples_tsv": str(task_dir / "samples.tsv"),
            "counts": {f"{split}_{label}": counts.get((split, str(label)), 0) for split in ["train", "val"] for label in [0, 1]},
        }
        (task_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append({"task_id": task_id, **manifest["counts"], "samples_tsv": str(task_dir / "samples.tsv")})
    write_tsv(out_dir / "dataset_summary.tsv", summary_rows, ["task_id", "train_0", "train_1", "val_0", "val_1", "samples_tsv"])
    (out_dir / "build_seen_counts.json").write_text(json.dumps({str(k): v for k, v in seen.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "tasks": len(summary_rows), "out_dir": str(out_dir), "summary": summary_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
