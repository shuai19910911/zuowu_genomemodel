#!/usr/bin/env python3
"""Extract frozen CropGenome-FM embeddings for formal CropGenome-Bench v1."""

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_server_transfer.downstream_v4.streaming_embeddings import (
    DiskBackedEmbeddingAccumulator,
)


def load_train_module(project_root: Path):
    path = project_root / "training_server_transfer" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("cropgenome_train_formal_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_label(row):
    if "label_json" in row and row["label_json"] != "":
        value = json.loads(row["label_json"])
        if isinstance(value, list):
            return [float(item) for item in value]
        return int(value) if isinstance(value, int) else float(value)
    return int(row["label"])


def labels_array(samples):
    values = [row["label"] for row in samples]
    if values and all(isinstance(value, int) for value in values):
        return np.asarray(values, dtype=np.int8)
    return np.asarray(values, dtype=np.float32)


EXPECTED_TASK_CACHE_FIELDS = {
    "embeddings", "sample_ids", "labels", "splits", "species", "assemblies",
    "group_ids", "rc_embeddings", "rc_sample_ids",
}


def validate_existing_task_cache(path, samples, include_rc_test):
    expected = {
        "sample_ids": np.asarray([row["sample_id"] for row in samples]),
        "labels": labels_array(samples),
        "splits": np.asarray([row["split"] for row in samples]),
        "species": np.asarray([row["species"] for row in samples]),
        "assemblies": np.asarray([
            row.get("assembly_id") or row["species"] for row in samples
        ]),
        "group_ids": np.asarray([
            row.get("group_id") or row.get("assembly_id") or row["species"]
            for row in samples
        ]),
        "rc_sample_ids": np.asarray([
            row["sample_id"] for row in samples if include_rc_test and row["split"] == "test"
        ]),
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != EXPECTED_TASK_CACHE_FIELDS:
            raise RuntimeError(f"task cache field set is invalid: {path}")
        for name, values in expected.items():
            observed = payload[name]
            equal = (
                np.array_equal(observed, values, equal_nan=True)
                if np.issubdtype(observed.dtype, np.number)
                and np.issubdtype(values.dtype, np.number)
                else np.array_equal(observed, values)
            )
            if not equal:
                raise RuntimeError(f"task cache {name} does not match frozen dataset: {path}")
        embeddings = payload["embeddings"]
        rc_embeddings = payload["rc_embeddings"]
        if embeddings.ndim != 2 or embeddings.shape[0] != len(samples) or embeddings.shape[1] <= 0:
            raise RuntimeError(f"task cache embeddings shape is invalid: {path}")
        if rc_embeddings.shape != (len(expected["rc_sample_ids"]), embeddings.shape[1]):
            raise RuntimeError(f"task cache RC embeddings shape is invalid: {path}")
        if not np.isfinite(embeddings).all() or not np.isfinite(rc_embeddings).all():
            raise RuntimeError(f"task cache contains non-finite embeddings: {path}")
        return int(embeddings.shape[1]), int(rc_embeddings.shape[0])


def read_samples(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for row in rows:
        input_path = Path(row["input_path"])
        if not input_path.is_absolute():
            input_path = path.parent / input_path
        row["input_path"] = str(input_path.resolve())
        row["offset"] = int(row["offset"])
        row["length"] = int(row["length"])
        row["label"] = parse_label(row)
    return rows


def read_sequence(row, mmap_cache, context, allow_shorter=False):
    path = row["input_path"]
    if path not in mmap_cache:
        mmap_cache[path] = np.memmap(path, dtype=np.uint8, mode="r")
    sequence = np.asarray(
        mmap_cache[path][row["offset"]: row["offset"] + row["length"]],
        dtype=np.int64,
    ).copy()
    if context > len(sequence) and not allow_shorter:
        raise ValueError(f"requested context {context} exceeds sample length {len(sequence)}")
    if context < len(sequence):
        start = (len(sequence) - context) // 2
        sequence = sequence[start:start + context]
    return sequence


def batchify(sequences, pad_id=6):
    max_len = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        tensor = torch.from_numpy(sequence).long()
        input_ids[index, :tensor.numel()] = tensor
        attention_mask[index, :tensor.numel()] = True
    return input_ids, attention_mask


def build_model(train_module, model_cfg, checkpoint_path, device):
    model_cfg = dict(model_cfg)
    model_cfg["gradient_checkpointing"] = False
    if checkpoint_path is None:
        raise RuntimeError("internal random-init evaluation is disabled by user")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = train_module.CropGenomeFM(model_cfg)
    load_report = train_module.load_model_state_safely(model, checkpoint["model"], allow_partial=False)
    checkpoint_step = int(checkpoint["step"])
    del checkpoint
    gc.collect()
    model.eval().to(device)
    if device.type == "cuda":
        model.half()
    return model, checkpoint_step, load_report, model_cfg


def pooled_embeddings(model, train_module, input_ids, attention_mask, device):
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"
    ):
        _, aux = model(input_ids, attention_mask, return_aux=True)
        pooled = train_module.CropGenomeFM.sequence_pool(aux["hidden"], attention_mask)
    return pooled.to(dtype=torch.float16).cpu().numpy()


def extract_task(model, train_module, samples, batch_size, context, device,
                 include_rc_test, allow_shorter=False, workspace=None):
    mmap_cache = {}
    rc_sample_ids = []
    rc_rows = sum(
        row["split"] == "test" for row in samples
    ) if include_rc_test else 0
    sink = DiskBackedEmbeddingAccumulator(
        workspace, forward_rows=len(samples), rc_rows=rc_rows,
    )
    started = time.time()
    try:
        for start in range(0, len(samples), batch_size):
            batch_rows = samples[start:start + batch_size]
            sequences = [read_sequence(row, mmap_cache, context, allow_shorter=allow_shorter) for row in batch_rows]
            input_ids, attention_mask = batchify(sequences)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            forward = pooled_embeddings(
                model, train_module, input_ids, attention_mask, device,
            )
            reverse = None
            if include_rc_test:
                test_indices = [index for index, row in enumerate(batch_rows) if row["split"] == "test"]
                if test_indices:
                    index_tensor = torch.tensor(test_indices, dtype=torch.long, device=device)
                    rc_ids = train_module.reverse_complement_tokens(input_ids.index_select(0, index_tensor))
                    rc_mask = attention_mask.index_select(0, index_tensor).flip(1)
                    reverse = pooled_embeddings(
                        model, train_module, rc_ids, rc_mask, device,
                    )
                    rc_sample_ids.extend(batch_rows[index]["sample_id"] for index in test_indices)
            sink.append(forward, reverse)
            del input_ids, attention_mask
    except Exception:
        sink.cleanup()
        raise
    return sink, np.asarray(rc_sample_ids), time.time() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-config", default="training_server_transfer/configs/model_large_v2.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--task", action="append", help="Only extract the named task; repeat for multiple tasks")
    parser.add_argument("--include-rc-test", action="store_true")
    parser.add_argument("--allow-shorter", action="store_true", help="Use a full sequence when shorter than --context")
    parser.add_argument("--resume-valid-task-caches", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    dataset_root = (project_root / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root).resolve()
    config_path = (project_root / args.model_config).resolve() if not Path(args.model_config).is_absolute() else Path(args.model_config).resolve()
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = (project_root / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    out_root = (project_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()
    out_dir = out_root / args.model_id / f"context_{args.context}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise SystemExit("CUDA requested but unavailable")
    train_module = load_train_module(project_root)
    model_cfg = load_json(config_path)
    model, checkpoint_step, load_report, effective_cfg = build_model(
        train_module, model_cfg, checkpoint_path, device
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    task_manifests = {}
    total_started = time.time()
    samples_paths = sorted(dataset_root.glob("*/samples.tsv"))
    if args.task:
        requested_tasks = set(args.task)
        samples_paths = [path for path in samples_paths if path.parent.name in requested_tasks]
        found_tasks = {path.parent.name for path in samples_paths}
        if found_tasks != requested_tasks:
            raise SystemExit(f"requested tasks not found: {sorted(requested_tasks - found_tasks)}")
    for samples_path in samples_paths:
        task_id = samples_path.parent.name
        samples = read_samples(samples_path)
        output_path = out_dir / f"{task_id}.npz"
        if args.resume_valid_task_caches and output_path.is_file():
            embedding_dim, rc_test_samples = validate_existing_task_cache(
                output_path, samples, args.include_rc_test,
            )
            task_manifests[task_id] = {
                "samples": len(samples), "embedding_dim": embedding_dim,
                "rc_test_samples": rc_test_samples, "elapsed_seconds": 0.0,
                "samples_sha256": sha256_path(samples_path),
                "cache_path": str(output_path), "cache_sha256": sha256_path(output_path),
                "reused_existing_cache": True,
            }
            print(json.dumps({
                "status": "task_reused", "model_id": args.model_id,
                "task_id": task_id, **task_manifests[task_id],
            }, ensure_ascii=False), flush=True)
            continue
        effective_batch_size = args.batch_size
        while True:
            try:
                sink, rc_sample_ids, elapsed = extract_task(
                    model, train_module, samples, effective_batch_size,
                    args.context, device, args.include_rc_test,
                    allow_shorter=args.allow_shorter,
                    workspace=output_path.with_name(output_path.name + ".parts"),
                )
                break
            except torch.cuda.OutOfMemoryError:
                if effective_batch_size <= args.min_batch_size:
                    raise
                effective_batch_size = max(
                    args.min_batch_size, effective_batch_size // 2,
                )
                torch.cuda.empty_cache()
                print(json.dumps({
                    "status": "oom_retry", "task_id": task_id,
                    "new_batch_size": effective_batch_size,
                }), flush=True)
        embedding_dim = sink.embedding_dim
        try:
            sink.save_npz(
                output_path,
            sample_ids=np.asarray([row["sample_id"] for row in samples]),
            labels=labels_array(samples),
            splits=np.asarray([row["split"] for row in samples]),
            species=np.asarray([row["species"] for row in samples]),
            assemblies=np.asarray([row.get("assembly_id") or row["species"] for row in samples]),
            group_ids=np.asarray([row.get("group_id") or row.get("assembly_id") or row["species"] for row in samples]),
            rc_sample_ids=rc_sample_ids,
            )
        except Exception:
            sink.cleanup()
            raise
        task_manifests[task_id] = {
            "samples": len(samples),
            "embedding_dim": int(embedding_dim),
            "rc_test_samples": int(len(rc_sample_ids)),
            "elapsed_seconds": elapsed,
            "samples_sha256": sha256_path(samples_path),
            "cache_path": str(output_path),
            "cache_sha256": sha256_path(output_path),
        }
        print(
            json.dumps(
                {"status": "task_ok", "model_id": args.model_id, "task_id": task_id, **task_manifests[task_id]},
                ensure_ascii=False,
            ),
            flush=True,
        )

    reused_tasks = []
    manifest_path = out_dir / "cache_manifest.json"
    if args.task and manifest_path.is_file():
        existing_manifest = load_json(manifest_path)
        for task_id, task_manifest in existing_manifest.get("tasks", {}).items():
            if task_id in task_manifests:
                continue
            samples_path = dataset_root / task_id / "samples.tsv"
            cache_path = Path(task_manifest["cache_path"])
            if not samples_path.is_file() or sha256_path(samples_path) != task_manifest["samples_sha256"]:
                raise SystemExit(f"cannot reuse {task_id}: dataset samples changed")
            if not cache_path.is_file() or sha256_path(cache_path) != task_manifest["cache_sha256"]:
                raise SystemExit(f"cannot reuse {task_id}: embedding cache changed")
            task_manifests[task_id] = task_manifest
            reused_tasks.append(task_id)

    manifest = {
        "status": "ok",
        "benchmark_id": "CropGenome-Bench-v1",
        "mode": "formal_frozen_embedding_cache",
        "model_id": args.model_id,
        "model_name": effective_cfg.get("model_name"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "internal_pretraining_ablations": "disabled_by_user",
        "model_config": str(config_path),
        "model_config_sha256": sha256_path(config_path),
        "implementation_sha256": sha256_path(Path(__file__).resolve()),
        "dataset_root": str(dataset_root),
        "context": args.context,
        "allow_shorter": args.allow_shorter,
        "batch_size": args.batch_size,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "loaded_keys": len(load_report["loaded_keys"]),
        "missing_keys": load_report["missing_keys"],
        "unexpected_keys": load_report["unexpected_keys"],
        "skipped_shape_mismatch": load_report["skipped_shape_mismatch"],
        "include_rc_test": args.include_rc_test,
        "selected_tasks": sorted(args.task) if args.task else "all",
        "reused_tasks": sorted(reused_tasks),
        "tasks": task_manifests,
        "elapsed_seconds": time.time() - total_started,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "manifest": manifest}, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
