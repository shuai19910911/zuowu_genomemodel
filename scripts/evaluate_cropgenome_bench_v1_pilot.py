#!/usr/bin/env python3
"""Evaluate CropGenome-Bench v1 pilot proxy tasks with a frozen checkpoint.

This evaluator supports the original pilot/validation datasets and a frozen
formal-lite train/test split produced from Stage_B region_bucket labels. It
reports simple sequence-composition baselines and a frozen CropGenome-FM
embedding nearest-centroid classifier. Stage_B proxy labels must still be
described as proxy evidence, not the final GFF-derived paper benchmark.
"""

import argparse
import csv
import gc
import importlib.util
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


METHOD_ORDER = ["majority_baseline", "one_mer_nearest_centroid", "model_embedding_nearest_centroid"]


def load_train_module(project_root: Path):
    train_path = project_root / "training_server_transfer" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("cropgenome_train", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def read_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_sequence(row, cache):
    input_path = row["input_path"]
    if input_path not in cache:
        cache[input_path] = np.memmap(input_path, dtype=np.uint8, mode="r")
    offset = int(row["offset"])
    length = int(row["length"])
    return np.asarray(cache[input_path][offset: offset + length], dtype=np.int64).copy()


def one_mer_features(sequences):
    feats = np.zeros((len(sequences), 4), dtype=np.float32)
    for i, seq in enumerate(sequences):
        valid = seq[seq < 4]
        if valid.size:
            counts = np.bincount(valid, minlength=4).astype(np.float32)
            feats[i] = counts / max(1.0, float(counts.sum()))
    return feats


def batchify(sequences, pad_id=6):
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        tensor = torch.from_numpy(seq).long()
        input_ids[i, : tensor.numel()] = tensor
        attention_mask[i, : tensor.numel()] = True
    return input_ids, attention_mask


def load_model(project_root: Path, checkpoint_path: Path, model_config_path: Path, device):
    train_module = load_train_module(project_root)
    model_cfg = load_json(model_config_path)
    model_cfg["gradient_checkpointing"] = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = train_module.CropGenomeFM(model_cfg)
    report = train_module.load_model_state_safely(model, checkpoint["model"], allow_partial=False)
    step = int(checkpoint["step"])
    del checkpoint
    gc.collect()
    model.eval().to(device)
    if device.type == "cuda":
        model.half()
    return model, train_module, step, report, model_cfg


def extract_embeddings(model, train_module, samples, batch_size, device):
    cache = {}
    sequences = []
    embeddings = []
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    for start in range(0, len(samples), batch_size):
        batch_rows = samples[start:start + batch_size]
        batch_sequences = [read_sequence(row, cache) for row in batch_rows]
        sequences.extend(batch_sequences)
        input_ids, attention_mask = batchify(batch_sequences)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            _, aux = model(input_ids, attention_mask, return_aux=True)
            pooled = train_module.CropGenomeFM.sequence_pool(aux["hidden"], attention_mask)
        embeddings.append(pooled.float().cpu().numpy())
        del input_ids, attention_mask, pooled, aux
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0), sequences


def nearest_centroid_predict(train_x, train_y, eval_x):
    classes = sorted(set(int(y) for y in train_y))
    centroids = []
    for cls in classes:
        centroids.append(train_x[np.asarray(train_y) == cls].mean(axis=0))
    centroids = np.asarray(centroids, dtype=np.float32)
    eval_norm = (eval_x ** 2).sum(axis=1, keepdims=True)
    centroid_norm = (centroids ** 2).sum(axis=1, keepdims=True).T
    distances = eval_norm + centroid_norm - 2.0 * eval_x @ centroids.T
    return np.asarray([classes[i] for i in distances.argmin(axis=1)], dtype=np.int64)


def majority_predict(train_y, n):
    counts = Counter(int(y) for y in train_y)
    majority = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return np.full(n, majority, dtype=np.int64)


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    mcc_den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn - fp * fn) / mcc_den) if mcc_den else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_tsv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_summary(rows, out_path: Path, title: str, footnote: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    tasks = sorted({r["task_id"] for r in rows})
    methods = ["majority_baseline", "one_mer_nearest_centroid", "model_embedding_nearest_centroid"]
    x = np.arange(len(tasks))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=180)
    colors = ["#9CA3AF", "#F59E0B", "#2563EB"]
    for i, method in enumerate(methods):
        vals = []
        for task in tasks:
            match = [r for r in rows if r["task_id"] == task and r["method"] == method]
            vals.append(float(match[0]["f1"]) if match else 0.0)
        ax.bar(x + (i - 1) * width, vals, width, label=method, color=colors[i])
    ax.set_title(title)
    ax.set_ylabel("F1 (pilot proxy; higher is better)")
    ax.set_xlabel("Pilot task")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, footnote, fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", facecolor="white")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root", default="training_server_transfer/runs/cropgenome_bench_v1_pilot/datasets")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default="training_server_transfer/configs/model_large.json")
    parser.add_argument("--output-root", default="training_server_transfer/runs/cropgenome_bench_v1_pilot/evaluations")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--benchmark-mode", default="pilot_proxy_from_stage_b_region_buckets")
    parser.add_argument("--result-note", default="pilot smoke test only; not formal CropGenome-Bench v1 paper result")
    parser.add_argument("--plot-title", default="CropGenome-Bench v1 proxy tasks — frozen embedding")
    parser.add_argument("--plot-footnote", default="Proxy labels from Stage_B region buckets; not final GFF-derived paper benchmark.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    dataset_root = (project_root / args.dataset_root).resolve()
    checkpoint_path = (project_root / args.checkpoint).resolve()
    model_config_path = (project_root / args.model_config).resolve()
    out_root = (project_root / args.output_root).resolve()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    model, train_module, checkpoint_step, load_report, model_cfg = load_model(project_root, checkpoint_path, model_config_path, device)
    all_rows = []
    confusion_rows = []
    for samples_path in sorted(dataset_root.glob("*/samples.tsv")):
        task_id = samples_path.parent.name
        samples = read_tsv(samples_path)
        train_samples = [r for r in samples if r["split"] == args.train_split]
        eval_samples = [r for r in samples if r["split"] == args.eval_split]
        if not train_samples or not eval_samples:
            continue
        train_emb, train_sequences = extract_embeddings(model, train_module, train_samples, args.batch_size, device)
        eval_emb, eval_sequences = extract_embeddings(model, train_module, eval_samples, args.batch_size, device)
        train_y = np.asarray([int(r["label"]) for r in train_samples], dtype=np.int64)
        eval_y = np.asarray([int(r["label"]) for r in eval_samples], dtype=np.int64)
        preds = {
            "majority_baseline": majority_predict(train_y, len(eval_y)),
            "one_mer_nearest_centroid": nearest_centroid_predict(one_mer_features(train_sequences), train_y, one_mer_features(eval_sequences)),
            "model_embedding_nearest_centroid": nearest_centroid_predict(train_emb.astype(np.float32), train_y, eval_emb.astype(np.float32)),
        }
        for method in METHOD_ORDER:
            metric = binary_metrics(eval_y, preds[method])
            all_rows.append({
                "checkpoint": checkpoint_path.stem,
                "checkpoint_step": checkpoint_step,
                "task_id": task_id,
                "mode": args.benchmark_mode,
                "method": method,
                "train_sample_count": len(train_samples),
                "eval_sample_count": len(eval_samples),
                **{k: f"{v:.8f}" if isinstance(v, float) else v for k, v in metric.items()},
                "note": args.result_note,
            })
            for true_label in [0, 1]:
                for pred_label in [0, 1]:
                    confusion_rows.append({
                        "checkpoint": checkpoint_path.stem,
                        "task_id": task_id,
                        "method": method,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": int(((eval_y == true_label) & (preds[method] == pred_label)).sum()),
                    })
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    eval_dir = out_root / checkpoint_path.stem
    source_dir = eval_dir / "source_data"
    metric_fields = [
        "checkpoint", "checkpoint_step", "task_id", "mode", "method", "train_sample_count", "eval_sample_count",
        "accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc", "tp", "tn", "fp", "fn", "note",
    ]
    write_tsv(source_dir / "pilot_metrics_summary.tsv", all_rows, metric_fields)
    write_tsv(source_dir / "pilot_confusion_matrix.tsv", confusion_rows, ["checkpoint", "task_id", "method", "true_label", "pred_label", "count"])
    figure_written = plot_summary(all_rows, eval_dir / "figures" / "pilot_task_f1.png", args.plot_title, args.plot_footnote)
    manifest = {
        "status": "ok",
        "benchmark_id": "CropGenome-Bench-v1",
        "mode": args.benchmark_mode,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_id": checkpoint_path.stem,
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dataset_root": str(dataset_root),
        "model_config": str(model_config_path),
        "model_name": model_cfg.get("model_name"),
        "loaded_keys": len(load_report["loaded_keys"]),
        "missing_keys": load_report["missing_keys"],
        "unexpected_keys": load_report["unexpected_keys"],
        "skipped_shape_mismatch": load_report["skipped_shape_mismatch"],
        "metrics_summary": str(source_dir / "pilot_metrics_summary.tsv"),
        "confusion_matrix": str(source_dir / "pilot_confusion_matrix.tsv"),
        "figure_written": figure_written,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "interpretation": args.result_note,
    }
    (eval_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "checkpoint_step": checkpoint_step, "rows": len(all_rows), "out_dir": str(eval_dir), "metrics": all_rows}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
