#!/usr/bin/env python3
"""Extract aligned per-token CropGenome-FM hidden states for B13 segmentation."""

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def sha256_path(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_score(seed, sample_id):
    payload = f"{int(seed)}\x1f{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_segmentation_samples(samples, max_per_group=8, seed=20260730):
    groups = {}
    for row in samples:
        key = (row["split"], row["assembly_id"], row["target_label"])
        groups.setdefault(key, []).append(row)
    selected = []
    for key in sorted(groups, key=lambda value: (SPLIT_ORDER[value[0]], value[1], value[2])):
        ordered = sorted(groups[key], key=lambda row: (stable_score(seed, row["sample_id"]), row["sample_id"]))
        selected.extend(ordered[: int(max_per_group)])
    return selected


def _read_slice(row, path_key, offset_key, length_key, dtype, context):
    values = np.memmap(row[path_key], dtype=dtype, mode="r")
    offset = int(row[offset_key])
    length = int(row[length_key])
    sequence = np.asarray(values[offset:offset + length], dtype=dtype).copy()
    if len(sequence) < int(context):
        raise RuntimeError(f"sample shorter than context: {row['sample_id']}")
    if len(sequence) > int(context):
        start = (len(sequence) - int(context)) // 2
        sequence = sequence[start:start + int(context)]
    return sequence


def extract_token_cache(model, samples, out_dir, context, batch_size, device, model_id):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    context = int(context)
    batch_size = int(batch_size)
    if not samples:
        raise RuntimeError("no segmentation samples selected")
    hidden_path = out_dir / "hidden.f16"
    labels_path = out_dir / "labels.u1"
    sample_path = out_dir / "selected_samples.tsv"
    labels_cache = np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(len(samples), context))
    hidden_cache = None
    hidden_dim = None
    finite_hidden = True
    started = time.time()
    for start in range(0, len(samples), batch_size):
        batch_rows = samples[start:start + batch_size]
        token_arrays = [
            _read_slice(row, "input_path", "offset", "length", np.uint8, context)
            for row in batch_rows
        ]
        label_arrays = [
            _read_slice(row, "label_path", "label_offset", "label_length", np.uint8, context)
            for row in batch_rows
        ]
        input_ids = torch.from_numpy(np.stack(token_arrays).astype(np.int64)).to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=device)
        amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"
        ):
            _, aux = model(input_ids, attention_mask, return_aux=True)
            hidden = aux["hidden"]
        hidden_np = hidden.float().cpu().numpy()
        if hidden_np.shape[:2] != (len(batch_rows), context):
            raise RuntimeError(f"unexpected hidden shape: {hidden_np.shape}")
        if not np.isfinite(hidden_np).all():
            finite_hidden = False
            raise RuntimeError("non-finite token hidden states")
        if hidden_cache is None:
            hidden_dim = int(hidden_np.shape[2])
            hidden_cache = np.memmap(
                hidden_path, dtype=np.float16, mode="w+",
                shape=(len(samples), context, hidden_dim),
            )
        elif int(hidden_np.shape[2]) != hidden_dim:
            raise RuntimeError("hidden dimension changed between batches")
        stop = start + len(batch_rows)
        hidden_cache[start:stop] = hidden_np.astype(np.float16)
        labels_cache[start:stop] = np.stack(label_arrays)
        del input_ids, attention_mask, hidden, hidden_np
    hidden_cache.flush()
    labels_cache.flush()
    fieldnames = [
        "sample_id", "split", "assembly_id", "species", "genus", "target_label",
        "feature_id", "contig_id", "center0", "strand",
    ]
    present = {key for row in samples for key in row}
    fieldnames = [key for key in fieldnames if key in present]
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    manifest = {
        "status": "ok",
        "task_id": "crop_gene_architecture_7state",
        "cache_type": "aligned_token_hidden_memmap",
        "model_id": model_id,
        "samples": len(samples),
        "context": context,
        "hidden_shape": [len(samples), context, hidden_dim],
        "labels_shape": [len(samples), context],
        "hidden_dtype": "float16",
        "labels_dtype": "uint8",
        "finite_hidden": finite_hidden,
        "hidden_path": str(hidden_path),
        "labels_path": str(labels_path),
        "selected_samples_path": str(sample_path),
        "hidden_sha256": sha256_path(hidden_path),
        "labels_sha256": sha256_path(labels_path),
        "selected_samples_sha256": sha256_path(sample_path),
        "elapsed_seconds": time.time() - started,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_samples(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        for key in ("input_path", "label_path"):
            value = Path(row[key])
            if not value.is_absolute():
                value = path.parent / value
            row[key] = str(value.resolve())
        for key in ("offset", "length", "label_offset", "label_length"):
            row[key] = int(row[key])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default="training_server_transfer/configs/model_large.json")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260730)

    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    samples_path = (root / args.samples).resolve() if not Path(args.samples).is_absolute() else Path(args.samples).resolve()
    checkpoint = None
    if args.checkpoint:
        checkpoint = (root / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint).resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint not found: {checkpoint}")
    model_config = (root / args.model_config).resolve() if not Path(args.model_config).is_absolute() else Path(args.model_config).resolve()
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise SystemExit("CUDA requested but unavailable")
    shared = _load_module(root / "scripts/extract_cropgenome_bench_v1_embeddings.py", "cropgenome_embedding_shared")
    train_module = shared.load_train_module(root)
    model_cfg = json.loads(model_config.read_text(encoding="utf-8"))
    model, checkpoint_step, load_report, _ = shared.build_model(
        train_module, model_cfg, checkpoint, device
    )
    selected = select_segmentation_samples(
        read_samples(samples_path), max_per_group=args.max_per_group, seed=args.seed
    )
    manifest = extract_token_cache(
        model, selected, output_dir, args.context, args.batch_size, device, args.model_id
    )
    manifest.update({
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": sha256_path(checkpoint),
        "internal_pretraining_ablations": "disabled_by_user",
        "model_config": str(model_config),
        "model_config_sha256": sha256_path(model_config),
        "samples_source": str(samples_path),
        "samples_source_sha256": sha256_path(samples_path),
        "selection_max_per_group": args.max_per_group,
        "selection_seed": args.seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "loaded_keys": len(load_report["loaded_keys"]),
        "missing_keys": load_report["missing_keys"],
        "unexpected_keys": load_report["unexpected_keys"],
        "implementation_sha256": sha256_path(Path(__file__).resolve()),
        "shared_extractor_sha256": sha256_path(root / "scripts/extract_cropgenome_bench_v1_embeddings.py"),
    })
    (output_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "manifest": manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
