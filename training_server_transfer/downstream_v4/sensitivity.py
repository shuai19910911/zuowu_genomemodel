"""B17 low-homology sensitivity analysis from sealed per-sample predictions."""

import csv
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from .data import read_canonical_rows, sha256_path
from .low_homology import LOW_HOMOLOGY_PROTOCOL_ID
from .metrics import evaluate_predictions
from .probe import _metrics as probe_metrics


PARENTS = ("B13", "B14", "B15", "B16")
SENSITIVITY_PROTOCOL_ID = "low_homology_seed_retention_v2"


def relative_retention(full_metric, low_metric):
    full_metric = float(full_metric); low_metric = float(low_metric)
    if not np.isfinite(full_metric) or not np.isfinite(low_metric) or full_metric == 0.0:
        raise ValueError("retention requires finite metrics and a non-zero full metric")
    return low_metric / full_metric


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _canonical_sha(payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def valid_sensitivity_receipt(row, final_root, plan_sha256):
    path = Path(final_root) / "results" / row["run_key"] / "ROW_RECEIPT.json"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    canonical = dict(receipt); stored = canonical.pop("receipt_sha256", None)
    if (
        receipt.get("status") != "ok"
        or receipt.get("run_key") != row["run_key"]
        or receipt.get("plan_sha256") != plan_sha256
        or receipt.get("analysis_protocol_id") != SENSITIVITY_PROTOCOL_ID
        or stored != _canonical_sha(canonical)
    ):
        return None
    for record in receipt.get("artifacts") or []:
        artifact = Path(record["path"])
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(record["size_bytes"])
            or sha256_path(artifact) != record["sha256"]
        ):
            return None
    return receipt


def _parent_run_key(row, task_id):
    scope = f"step{int(row['checkpoint_step']):05d}" if row["checkpoint_scope"] == "step" else "shared"
    return f"{scope}__{task_id}__{row['model_id']}__ctx{int(row['context_bp'])}"


def _cohort_ids(final_root, task_id):
    path = Path(final_root) / "low_homology" / f"{task_id}.low_homology.tsv"
    receipt = path.with_name(f"{task_id}.RECEIPT.json")
    if not path.is_file() or not receipt.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    canonical = dict(payload); stored = canonical.pop("receipt_sha256", None)
    hits_path = Path(payload.get("hits_path", ""))
    samples_path = Path(payload.get("samples_path", ""))
    if (
        payload.get("status") != "ok"
        or payload.get("task_id") != task_id
        or payload.get("protocol_id") != LOW_HOMOLOGY_PROTOCOL_ID
        or stored != _canonical_sha(canonical)
        or sha256_path(path) != payload.get("cohort_sha256")
        or not hits_path.is_file()
        or sha256_path(hits_path) != payload.get("hits_sha256")
        or not samples_path.is_file()
        or sha256_path(samples_path) != payload.get("samples_sha256")
    ):
        raise RuntimeError(f"invalid low-homology receipt for {task_id}")
    ids = set(); cohort_sample_ids = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            cohort_sample_ids.append(row["sample_id"])
            if int(row["low_homology"]):
                ids.add(row["sample_id"])
    test_ids = {
        row["sample_id"] for row in read_canonical_rows(samples_path)
        if row["split"] == "test"
    }
    if (
        len(cohort_sample_ids) != int(payload.get("test_samples", -1))
        or len(set(cohort_sample_ids)) != len(cohort_sample_ids)
        or set(cohort_sample_ids) != test_ids
        or len(ids) != int(payload.get("low_homology_samples", -1))
    ):
        raise RuntimeError(f"low-homology cohort/sample mismatch for {task_id}")
    if not ids:
        raise RuntimeError(f"empty low-homology cohort for {task_id}")
    return ids, path, receipt


def _mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    margin = (
        float(student_t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
        * float(values.std(ddof=1)) / np.sqrt(len(values))
    )
    return mean - margin, mean + margin


def _metric_pair(task, labels, scores, predictions, groups, selected):
    if task["task_id"] == "B13":
        full = evaluate_predictions(
            "token_multiclass", labels, predictions=predictions, boundary_tolerance=20,
        )
        low = evaluate_predictions(
            "token_multiclass", labels[selected], predictions=predictions[selected],
            boundary_tolerance=20,
        )
    else:
        full = probe_metrics(task["task_kind"], labels, scores, predictions, groups)
        low = probe_metrics(
            task["task_kind"], labels[selected], scores[selected],
            predictions[selected], groups[selected],
        )
    return full, low


def _task_metrics(task, prediction_path, low_ids):
    with np.load(prediction_path, allow_pickle=False) as prediction:
        sample_ids = prediction["sample_ids"].astype(str)
        selected = np.asarray([sample_id in low_ids for sample_id in sample_ids], dtype=bool)
        if not np.any(selected):
            raise RuntimeError(f"prediction/cohort sample IDs do not overlap for {task['task_id']}")
        labels = prediction["labels"]
        scores = prediction["scores"].astype(np.float32)
        predictions = prediction["predictions"]
        groups = None if task["task_id"] == "B13" else prediction["group_ids"].astype(str)
        full, low = _metric_pair(task, labels, scores, predictions, groups, selected)
        seeds = prediction["seeds"].astype(int)
        seed_predictions = prediction["seed_predictions"]
        seed_scores = None if task["task_id"] == "B13" else prediction["seed_scores"].astype(np.float32)
        if len(seeds) != 5 or seed_predictions.shape[0] != 5:
            raise RuntimeError(f"{task['task_id']} prediction lacks five sealed seed outputs")
        primary = task["primary_metric"]
        seed_records = []
        for index, seed in enumerate(seeds):
            one_scores = scores if seed_scores is None else seed_scores[index]
            one_full, one_low = _metric_pair(
                task, labels, one_scores, seed_predictions[index], groups, selected,
            )
            seed_records.append({
                "seed": int(seed), "full_metric": float(one_full[primary]),
                "low_homology_metric": float(one_low[primary]),
                "relative_metric_retention": relative_retention(
                    one_full[primary], one_low[primary],
                ),
            })
    if primary not in full or primary not in low:
        raise RuntimeError(f"primary metric {primary} missing for {task['task_id']}")
    retention_values = [record["relative_metric_retention"] for record in seed_records]
    ci_low, ci_high = _mean_ci(retention_values)
    return {
        "task_id": task["task_id"], "primary_metric": primary,
        "full_metric": float(full[primary]), "low_homology_metric": float(low[primary]),
        "relative_metric_retention": relative_retention(full[primary], low[primary]),
        "seed_retention_mean": float(np.mean(retention_values)),
        "seed_retention_std": float(np.std(retention_values, ddof=1)),
        "seed_retention_ci95_low": ci_low, "seed_retention_ci95_high": ci_high,
        "seed_metrics": seed_records,
        "full_samples": int(len(sample_ids)), "low_homology_samples": int(selected.sum()),
        "low_homology_fraction": float(selected.mean()),
        "cohort_equals_full_test": bool(selected.all()),
        "sensitivity_informative": not bool(selected.all()),
    }


def sensitivity_readiness(row, registry, final_root):
    if row.get("task_id") != "B17":
        return False, "not_b17"
    final_root = Path(final_root)
    for task_id in PARENTS:
        parent = _parent_run_key(row, task_id)
        prediction_path = final_root / "results" / parent / "test_predictions.npz"
        if not prediction_path.is_file():
            return False, f"prediction:{task_id}"
        try:
            with np.load(prediction_path, allow_pickle=False) as prediction:
                required = {"sample_ids", "labels", "scores", "predictions", "seeds", "seed_predictions"}
                if task_id != "B13":
                    required |= {"group_ids", "seed_scores"}
                if required - set(prediction.files) or len(prediction["seeds"]) != 5:
                    return False, f"prediction_protocol:{task_id}"
            _cohort_ids(final_root, task_id)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            return False, f"cohort_protocol:{task_id}"
    return True, "ready"


def execute_sensitivity_row(row, registry, final_root, plan_sha256):
    final_root = Path(final_root).resolve()
    resumed = valid_sensitivity_receipt(row, final_root, plan_sha256)
    if resumed:
        return {**resumed, "resumed": True}
    ready, reason = sensitivity_readiness(row, registry, final_root)
    if not ready:
        raise RuntimeError(f"B17 row is not ready: {reason}")
    task_map = {task["task_id"]: task for task in registry["tasks"]}
    output = final_root / "results" / row["run_key"]
    output.mkdir(parents=True, exist_ok=True)
    records = []; artifacts = []
    for task_id in PARENTS:
        parent = _parent_run_key(row, task_id)
        prediction_path = final_root / "results" / parent / "test_predictions.npz"
        low_ids, cohort_path, cohort_receipt = _cohort_ids(final_root, task_id)
        records.append(_task_metrics(task_map[task_id], prediction_path, low_ids))
        for path in (prediction_path, cohort_path, cohort_receipt):
            artifacts.append({"path": str(path), "sha256": sha256_path(path), "size_bytes": path.stat().st_size})
    source_path = output / "source_data.tsv"
    summary_records = [
        {key: value for key, value in record.items() if key != "seed_metrics"}
        for record in records
    ]
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_records[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(summary_records)
    seed_source_path = output / "seed_source_data.tsv"
    seed_rows = [
        {"task_id": record["task_id"], "primary_metric": record["primary_metric"], **seed_record}
        for record in records for seed_record in record["seed_metrics"]
    ]
    with seed_source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(seed_rows)
    ordered_seeds = sorted({item["seed"] for record in records for item in record["seed_metrics"]})
    per_seed_means = [
        float(np.mean([
            next(item for item in record["seed_metrics"] if item["seed"] == seed)["relative_metric_retention"]
            for record in records
        ]))
        for seed in ordered_seeds
    ]
    overall_ci_low, overall_ci_high = _mean_ci(per_seed_means)
    metrics = {
        "mean_relative_metric_retention": float(np.mean([record["relative_metric_retention"] for record in records])),
        "relative_metric_retention": float(np.mean([record["relative_metric_retention"] for record in records])),
        "min_relative_metric_retention": float(np.min([record["relative_metric_retention"] for record in records])),
        "seed_mean_relative_metric_retention": float(np.mean(per_seed_means)),
        "seed_mean_retention_ci95_low": overall_ci_low,
        "seed_mean_retention_ci95_high": overall_ci_high,
        "seeds": ordered_seeds,
        "parent_tasks": records,
        "all_cohorts_equal_full_test": all(record["cohort_equals_full_test"] for record in records),
        "limitation": (
            "all_low_homology_cohorts_equal_the_full_test_sets"
            if all(record["cohort_equals_full_test"] for record in records) else None
        ),
    }
    metrics_path = output / "metrics.json"; _atomic_json(metrics_path, metrics)
    for path in (source_path, seed_source_path, metrics_path):
        artifacts.append({"path": str(path), "sha256": sha256_path(path), "size_bytes": path.stat().st_size})
    receipt = {
        "status": "ok", "run_key": row["run_key"], "plan_sha256": plan_sha256,
        "analysis_protocol_id": SENSITIVITY_PROTOCOL_ID,
        "row": row, "metrics": metrics, "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    receipt_path = output / "ROW_RECEIPT.json"; _atomic_json(receipt_path, receipt)
    return receipt


def execute_ready_sensitivity_rows(rows, registry, final_root, plan_sha256, max_rows=32):
    from .final_controller import _valid_terminal_cpu_failure, record_cpu_row_failure

    completed = []; failures = []; waiting = {}; already_complete = 0
    for row in rows:
        if row.get("task_id") != "B17":
            continue
        if valid_sensitivity_receipt(row, final_root, plan_sha256):
            already_complete += 1
            continue
        if _valid_terminal_cpu_failure(row, final_root, plan_sha256):
            continue
        ready, reason = sensitivity_readiness(row, registry, final_root)
        if not ready:
            waiting[reason] = waiting.get(reason, 0) + 1
            continue
        try:
            completed.append(execute_sensitivity_row(row, registry, final_root, plan_sha256))
        except Exception:
            failure = record_cpu_row_failure(
                row, final_root, plan_sha256, traceback.format_exc(),
            )
            failures.append({"run_key": row["run_key"], "status": failure["status"]})
        if len(completed) + len(failures) >= int(max_rows):
            break
    return {
        "status": "ok", "completed": len(completed),
        "already_complete": already_complete, "failures": failures, "waiting": waiting,
    }
