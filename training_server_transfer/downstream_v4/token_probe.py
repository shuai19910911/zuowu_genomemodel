"""Streaming linear token probe for B13 aligned hidden-state caches."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from .data import sha256_path
from .feature_projection import (
    HEAD_INPUT_DIM, PROJECTION_PROTOCOL_ID, projection_spec,
)
from .metrics import evaluate_predictions


TOKEN_PROBE_PROTOCOL_ID = "seeded_hash256_token_linear_probe_v2"


def _resolve(cache_dir, value):
    path = Path(value)
    return path if path.is_absolute() else Path(cache_dir) / path


def _read_samples(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    splits = np.asarray([
        "validation" if row["split"] in {"val", "dev"} else row["split"]
        for row in rows
    ])
    return splits, np.asarray([row["sample_id"] for row in rows])


def _project_hidden(values, buckets, scales):
    projected = torch.zeros(
        (*values.shape[:-1], HEAD_INPUT_DIM), dtype=values.dtype, device=values.device,
    )
    projected.index_add_(-1, buckets, values * scales)
    return projected


def _predict(model, hidden, indices, batch_samples, device, n_classes, buckets, scales):
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_samples):
            batch_index = indices[start:start + batch_samples]
            values = torch.from_numpy(np.asarray(hidden[batch_index], dtype=np.float32)).to(device)
            values = _project_hidden(values, buckets, scales)
            logits = model(values)
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    if not probabilities:
        return np.empty((0, hidden.shape[1], n_classes), dtype=np.float32)
    return np.concatenate(probabilities, axis=0)


def evaluate_token_cache(cache_dir, output_dir, seeds=(13, 29, 43, 71, 97), max_epochs=80,
                         patience=12, batch_samples=4, learning_rate=1e-3,
                         weight_decay=1e-4, boundary_tolerance=20, device="cuda"):
    cache_dir = Path(cache_dir).resolve(); output_dir = Path(output_dir).resolve()
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok": raise ValueError("token cache manifest is not ok")
    hidden_shape = tuple(int(x) for x in manifest["hidden_shape"])
    labels_shape = tuple(int(x) for x in manifest["labels_shape"])
    if hidden_shape[:2] != labels_shape: raise ValueError("hidden/label shape mismatch")
    hidden_path = _resolve(cache_dir, manifest["hidden_path"])
    labels_path = _resolve(cache_dir, manifest["labels_path"])
    samples_path = _resolve(cache_dir, manifest["selected_samples_path"])
    hidden = np.memmap(hidden_path, dtype=np.float16, mode="r", shape=hidden_shape)
    labels = np.memmap(labels_path, dtype=np.uint8, mode="r", shape=labels_shape)
    splits, sample_ids = _read_samples(samples_path)
    train_idx = np.flatnonzero(splits == "train"); val_idx = np.flatnonzero(splits == "validation"); test_idx = np.flatnonzero(splits == "test")
    if not len(train_idx) or not len(val_idx) or not len(test_idx): raise ValueError("token cache needs train/validation/test")
    observed = np.asarray(labels[np.concatenate([train_idx, val_idx])]).reshape(-1)
    observed = observed[observed != 255]
    n_classes = int(observed.max()) + 1
    test_observed = np.asarray(labels[test_idx]).reshape(-1)
    test_observed = test_observed[test_observed != 255]
    if np.any(test_observed >= n_classes): raise ValueError("test contains unseen token class")
    train_observed = np.asarray(labels[train_idx]).reshape(-1)
    train_observed = train_observed[train_observed != 255]
    counts = np.bincount(train_observed, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    class_weights = counts.sum() / (n_classes * counts)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    bucket_values, scale_values = projection_spec(hidden_shape[2])
    projection_buckets = torch.as_tensor(bucket_values, dtype=torch.long, device=torch_device)
    projection_scales = torch.as_tensor(scale_values, dtype=torch.float32, device=torch_device)
    selections = []; test_probabilities = []
    for seed in seeds:
        torch.manual_seed(int(seed)); np.random.seed(int(seed))
        model = torch.nn.Linear(HEAD_INPUT_DIM, n_classes).to(torch_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=torch_device), ignore_index=255)
        best_state = None; best_f1 = -1.0; stale = 0; history = []
        generator = np.random.default_rng(int(seed))
        for epoch in range(1, int(max_epochs) + 1):
            model.train(); order = generator.permutation(train_idx); total_loss = 0.0; batches = 0
            for start in range(0, len(order), int(batch_samples)):
                idx = order[start:start + int(batch_samples)]
                x = torch.from_numpy(np.asarray(hidden[idx], dtype=np.float32)).to(torch_device)
                x = _project_hidden(x, projection_buckets, projection_scales)
                y = torch.from_numpy(np.asarray(labels[idx], dtype=np.int64)).to(torch_device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x).reshape(-1, n_classes), y.reshape(-1))
                loss.backward(); optimizer.step(); total_loss += float(loss.detach().cpu()); batches += 1
            val_probs = _predict(
                model, hidden, val_idx, batch_samples, torch_device, n_classes,
                projection_buckets, projection_scales,
            )
            val_pred = val_probs.argmax(axis=-1); val_truth = np.asarray(labels[val_idx])
            valid = val_truth.reshape(-1) != 255
            score = float(f1_score(
                val_truth.reshape(-1)[valid], val_pred.reshape(-1)[valid],
                average="macro", zero_division=0,
            ))
            history.append({"epoch": epoch, "train_loss": total_loss / max(1, batches), "validation_macro_f1": score})
            if score > best_f1 + 1e-8:
                best_f1 = score; stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= int(patience): break
        model.load_state_dict(best_state)
        test_probabilities.append(_predict(
            model, hidden, test_idx, batch_samples, torch_device, n_classes,
            projection_buckets, projection_scales,
        ))
        selections.append({"seed": int(seed), "best_validation_macro_f1": best_f1, "epochs": len(history), "history": history})
    mean_probability = np.mean(test_probabilities, axis=0)
    test_prediction = mean_probability.argmax(axis=-1); test_truth = np.asarray(labels[test_idx])
    metrics = evaluate_predictions(
        "token_multiclass", test_truth, predictions=test_prediction,
        boundary_tolerance=boundary_tolerance,
    )
    seed_test_metrics = [
        {
            "seed": int(seed),
            "metrics": evaluate_predictions(
                "token_multiclass", test_truth,
                predictions=probability.argmax(axis=-1),
                boundary_tolerance=boundary_tolerance,
            ),
        }
        for seed, probability in zip(seeds, test_probabilities)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "test_predictions.npz"
    np.savez_compressed(
        prediction_path, sample_ids=sample_ids[test_idx], labels=test_truth,
        scores=mean_probability.astype(np.float16), predictions=test_prediction,
        seeds=np.asarray(seeds, dtype=np.int64),
        seed_predictions=np.stack([
            probability.argmax(axis=-1) for probability in test_probabilities
        ]).astype(np.uint8),
    )
    receipt = {
        "status": "ok", "task_id": "B13", "task_kind": "token_multiclass",
        "probe_protocol_id": TOKEN_PROBE_PROTOCOL_ID,
        "head_training": {
            "optimizer": "AdamW", "random_initialisation": True,
            "seeded_train_shuffle": True, "validation_early_stopping": True,
            "feature_projection": PROJECTION_PROTOCOL_ID,
            "head_input_dim": HEAD_INPUT_DIM,
        },
        "cache_manifest": str(manifest_path), "cache_manifest_sha256": sha256_path(manifest_path),
        "hidden_sha256": sha256_path(hidden_path), "labels_sha256": sha256_path(labels_path),
        "hidden_shape": list(hidden_shape), "source_embedding_dim": int(hidden_shape[2]),
        "embedding_dim": HEAD_INPUT_DIM, "n_classes": n_classes,
        "seeds": [int(seed) for seed in seeds], "selection": selections,
        "test_metrics": metrics, "test_access_count": 1,
        "seed_test_metrics": seed_test_metrics,
        "test_predictions": str(prediction_path),
        "test_predictions_sha256": sha256_path(prediction_path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    canonical = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    receipt["receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    (output_dir / "FINAL_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt
