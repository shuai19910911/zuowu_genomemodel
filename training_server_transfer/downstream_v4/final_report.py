"""Artifact-only statistical aggregation for the closed downstream matrix."""

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from .data import sha256_path
from .final_audit import audit_final_closure
from .final_controller import (
    _valid_cpu_row_receipt, _valid_group_receipt,
    _valid_terminal_cpu_failure, _valid_terminal_group_failure,
    build_gpu_groups,
)
from .sensitivity import valid_sensitivity_receipt


REPORT_PROTOCOL_ID = "artifact_only_paired_seed_and_bootstrap_final_report_v2"


def _canonical_sha(payload):
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_hashed_receipt(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = dict(payload); stored = canonical.pop("receipt_sha256", None)
    if payload.get("status") != "ok" or stored != _canonical_sha(canonical):
        raise RuntimeError(f"invalid result receipt: {path}")
    return payload


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        value = float(values[0]) if len(values) else None
        return value, value
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean - half, mean + half


def _seed_values(receipt, row):
    primary = row["primary_metric"]
    if row["execution_kind"] == "analysis":
        metrics = receipt["metrics"]
        parents = metrics.get("parent_tasks") or []
        seeds = [int(seed) for seed in metrics.get("seeds") or []]
        result = {}
        for seed in seeds:
            values = [
                next(
                    item["relative_metric_retention"]
                    for item in parent["seed_metrics"] if int(item["seed"]) == seed
                )
                for parent in parents
            ]
            result[seed] = float(np.mean(values))
        return result
    result = {}
    for record in receipt.get("seed_test_metrics") or []:
        if primary not in record.get("metrics", {}):
            raise RuntimeError(f"seed result lacks primary metric {primary}: {row['run_key']}")
        result[int(record["seed"])] = float(record["metrics"][primary])
    return result


def _pairing_tokens(receipt, row):
    seeds = [int(seed) for seed in receipt.get("seeds") or []]
    if row["execution_kind"] == "analysis":
        seeds = [int(seed) for seed in receipt["metrics"].get("seeds") or []]
        return {seed: f"sensitivity-fixed-cohorts:{seed}" for seed in seeds}
    if row["task_kind"] == "token_multiclass":
        return {seed: f"token-fixed-train-split:{seed}" for seed in seeds}
    if row["task_kind"].startswith("zero_shot"):
        return {}
    if receipt.get("folds"):
        by_seed = {seed: [] for seed in seeds}
        for fold in receipt["folds"]:
            held_out = str(fold["held_out_group"])
            for parameter in fold["parameters"]:
                seed = int(parameter["seed"])
                by_seed[seed].append((
                    held_out, parameter["final_train_bootstrap"]["indices_sha256"],
                ))
        return {seed: _canonical_sha(values) for seed, values in by_seed.items()}
    return {
        int(record["seed"]): record["final_train_bootstrap"]["indices_sha256"]
        for record in receipt.get("selection") or []
    }


def _result_receipt(row, final_root):
    if row["task_kind"].startswith("zero_shot"):
        path = Path(final_root) / "zero_shot" / row["run_key"] / "FINAL_RECEIPT.json"
    else:
        path = Path(final_root) / "results" / row["run_key"] / (
            "ROW_RECEIPT.json" if row["execution_kind"] == "analysis" else "FINAL_RECEIPT.json"
        )
    receipt = _read_hashed_receipt(path)
    metrics = (
        receipt["metrics"]
        if row["execution_kind"] == "analysis" or row["task_kind"].startswith("zero_shot")
        else receipt["test_metrics"]
    )
    primary = row["primary_metric"]
    if primary not in metrics:
        raise RuntimeError(f"result lacks primary metric {primary}: {row['run_key']}")
    value = float(metrics[primary])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite primary metric: {row['run_key']}")
    seeds = _seed_values(receipt, row)
    if not row["task_kind"].startswith("zero_shot") and len(seeds) != 5:
        raise RuntimeError(f"trained-head row lacks five seed metrics: {row['run_key']}")
    bootstrap_values = {}
    bootstrap_token = None
    bootstrap = receipt.get("bootstrap") or {}
    if bootstrap:
        source = Path(bootstrap.get("source_data", ""))
        if not source.is_file() or sha256_path(source) != bootstrap.get("source_data_sha256"):
            raise RuntimeError(f"invalid zero-shot bootstrap source: {row['run_key']}")
        with source.open("r", encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                bootstrap_values[int(record["replicate"])] = float(record["value"])
        if len(bootstrap_values) != int(bootstrap.get("replicates", -1)):
            raise RuntimeError(f"incomplete zero-shot bootstrap source: {row['run_key']}")
        bootstrap_token = bootstrap.get("resample_indices_sha256")
    return (
        receipt, value, seeds, _pairing_tokens(receipt, row),
        bootstrap_values, bootstrap_token,
    )


def _is_cpu_row(row):
    return (
        row["execution_kind"] == "evaluation"
        and row["task_kind"] != "token_multiclass"
        and not row["task_kind"].startswith("zero_shot")
    )


def _failure_reason(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "terminal_failure_receipt_unreadable"
    return str(payload.get("reason") or "terminal_failure")


def _write_tsv(path, rows, fieldnames):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _holm_adjust(records):
    indexed = [(index, row["paired_p_value"]) for index, row in enumerate(records) if row["paired_p_value"] is not None]
    indexed.sort(key=lambda item: item[1])
    running = 0.0; total = len(indexed)
    adjusted = {}
    for rank, (index, value) in enumerate(indexed, start=1):
        candidate = min(1.0, float(value) * (total - rank + 1))
        running = max(running, candidate); adjusted[index] = running
    for index, row in enumerate(records):
        row["holm_adjusted_p"] = adjusted.get(index)


def build_final_report(plan, registry, project_root, final_root):
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    audit = audit_final_closure(plan, registry, project_root, final_root, deep=True)
    if audit["status"] not in {"ok", "closed_with_failures"}:
        raise RuntimeError(f"final matrix is not closed: {audit['blockers']}")
    plan_sha = plan["plan_sha256"]
    groups = build_gpu_groups(plan)
    group_by_row = {run_key: group for group in groups for run_key in group["run_keys"]}
    group_ok = {
        group["group_key"]: bool(_valid_group_receipt(group, final_root, plan_sha))
        for group in groups
    }
    group_terminal = {
        group["group_key"]: bool(_valid_terminal_group_failure(group, final_root, plan_sha))
        for group in groups
    }
    result_rows = []; seed_rows = []; bootstrap_rows = []; status_rows = []; successful = {}
    low_homology_equals_full_test = False
    for row in plan["rows"]:
        key = row["run_key"]
        cpu_ok = bool(_valid_cpu_row_receipt(row, final_root, plan_sha)) if _is_cpu_row(row) else False
        cpu_terminal = bool(_valid_terminal_cpu_failure(row, final_root, plan_sha))
        group = group_by_row.get(key); terminal_path = None
        if row["execution_kind"] == "analysis":
            success = bool(valid_sensitivity_receipt(row, final_root, plan_sha))
            terminal = cpu_terminal
            if terminal:
                terminal_path = final_root / "results" / key / "TERMINAL_FAILURE.json"
        elif row["model_kind"] == "simple_baseline":
            success = cpu_ok; terminal = cpu_terminal
            if terminal:
                terminal_path = final_root / "results" / key / "TERMINAL_FAILURE.json"
        elif _is_cpu_row(row):
            success = bool(group and group_ok[group["group_key"]] and cpu_ok)
            terminal = bool(group and group_terminal[group["group_key"]]) or cpu_terminal
            if group and group_terminal[group["group_key"]]:
                terminal_path = final_root / "gpu_groups" / group["group_key"] / "FAILED.json"
            elif cpu_terminal:
                terminal_path = final_root / "results" / key / "TERMINAL_FAILURE.json"
        else:
            success = bool(group and group_ok[group["group_key"]])
            terminal = bool(group and group_terminal[group["group_key"]])
            if terminal:
                terminal_path = final_root / "gpu_groups" / group["group_key"] / "FAILED.json"
        if success:
            (
                receipt, value, seeds, pairing_tokens,
                bootstrap_values, bootstrap_token,
            ) = _result_receipt(row, final_root)
            if (
                row["execution_kind"] == "analysis"
                and (receipt.get("metrics") or {}).get("all_cohorts_equal_full_test") is True
            ):
                low_homology_equals_full_test = True
            seed_values = list(seeds.values())
            ci_low, ci_high = _mean_ci(seed_values)
            if seed_values:
                uncertainty_method = "student_t_across_five_paired_seeds"
                uncertainty_low, uncertainty_high = ci_low, ci_high
            elif bootstrap_values:
                uncertainty_method = "sealed_test_stratified_bootstrap_percentile"
                uncertainty_low = float(np.quantile(list(bootstrap_values.values()), 0.025))
                uncertainty_high = float(np.quantile(list(bootstrap_values.values()), 0.975))
            else:
                uncertainty_method = "none"
                uncertainty_low = uncertainty_high = None
            result = {
                "task_id": row["task_id"], "model_id": row["model_id"],
                "checkpoint_step": row.get("checkpoint_step"), "context_bp": row["context_bp"],
                "primary_metric": row["primary_metric"], "primary_value": value,
                "seed_mean": float(np.mean(seed_values)) if seed_values else None,
                "seed_std": float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else None,
                "seed_ci95_low": ci_low if seed_values else None,
                "seed_ci95_high": ci_high if seed_values else None,
                "seed_count": len(seed_values),
                "uncertainty_method": uncertainty_method,
                "ci95_low": uncertainty_low, "ci95_high": uncertainty_high,
                "bootstrap_replicates": len(bootstrap_values), "run_key": key,
            }
            result_rows.append(result); successful[key] = {
                "row": row, "value": value, "seeds": seeds,
                "pairing_tokens": pairing_tokens,
                "bootstrap": bootstrap_values, "bootstrap_token": bootstrap_token,
            }
            seed_rows.extend({
                "task_id": row["task_id"], "model_id": row["model_id"],
                "checkpoint_step": row.get("checkpoint_step"), "context_bp": row["context_bp"],
                "primary_metric": row["primary_metric"], "seed": seed,
                "value": seed_value, "run_key": key,
            } for seed, seed_value in sorted(seeds.items()))
            bootstrap_rows.extend({
                "task_id": row["task_id"], "model_id": row["model_id"],
                "checkpoint_step": row.get("checkpoint_step"), "context_bp": row["context_bp"],
                "primary_metric": row["primary_metric"], "replicate": replicate,
                "value": replicate_value, "resample_indices_sha256": bootstrap_token,
                "run_key": key,
            } for replicate, replicate_value in sorted(bootstrap_values.items()))
            status = "complete"; reason = ""
        elif terminal:
            status = "terminal_failed"; reason = _failure_reason(terminal_path)
        else:
            raise RuntimeError(f"audit/report state disagreement for {key}")
        status_rows.append({
            "cell_type": "applicable_row", "task_id": row["task_id"],
            "model_id": row["model_id"], "context_bp": row["context_bp"],
            "checkpoint_step": row.get("checkpoint_step"), "status": status,
            "reason": reason, "cell_id": key,
        })
    for item in plan["applicability"]:
        if item["status"] == "not_applicable":
            cell_id = f"NA__{item['task_id']}__{item['model_id']}"
            status_rows.append({
                "cell_type": "not_applicable", "task_id": item["task_id"],
                "model_id": item["model_id"], "context_bp": "",
                "checkpoint_step": "", "status": "not_applicable",
                "reason": item["reason"], "cell_id": cell_id,
            })
    comparisons = []
    ours = [item for item in successful.values() if item["row"]["model_id"] == "CropGenomeFM" and item["row"].get("checkpoint_step") == 50000]
    for our in ours:
        row = our["row"]
        comparators = [
            item for item in successful.values()
            if item["row"]["task_id"] == row["task_id"]
            and item["row"]["context_bp"] == row["context_bp"]
            and item["row"]["checkpoint_scope"] == "shared"
            and item["row"]["model_id"] != "CropGenomeFM"
        ]
        for baseline in comparators:
            common = sorted(
                seed for seed in set(our["seeds"]) & set(baseline["seeds"])
                if our["pairing_tokens"].get(seed) == baseline["pairing_tokens"].get(seed)
                and our["pairing_tokens"].get(seed) is not None
            )
            deltas = [our["seeds"][seed] - baseline["seeds"][seed] for seed in common]
            if len(common) == 5:
                inference_method = "paired_seed_student_t"
                paired_bootstrap_count = 0
                delta_mean = float(np.mean(deltas)); delta_std = float(np.std(deltas, ddof=1))
                ci_low, ci_high = _mean_ci(deltas)
                if delta_std == 0.0:
                    p_value = 1.0 if delta_mean == 0.0 else 0.0
                else:
                    statistic = delta_mean / (delta_std / math.sqrt(len(deltas)))
                    p_value = float(2.0 * student_t.sf(abs(statistic), len(deltas) - 1))
            elif (
                our.get("bootstrap_token") is not None
                and our.get("bootstrap_token") == baseline.get("bootstrap_token")
                and set(our.get("bootstrap") or {}) == set(baseline.get("bootstrap") or {})
                and len(our.get("bootstrap") or {}) >= 100
            ):
                inference_method = "paired_sealed_test_bootstrap"
                bootstrap_replicates = sorted(our["bootstrap"])
                bootstrap_deltas = np.asarray([
                    our["bootstrap"][replicate] - baseline["bootstrap"][replicate]
                    for replicate in bootstrap_replicates
                ], dtype=float)
                paired_bootstrap_count = len(bootstrap_replicates)
                delta_mean = float(our["value"] - baseline["value"])
                delta_std = float(np.std(bootstrap_deltas, ddof=1))
                ci_low = float(np.quantile(bootstrap_deltas, 0.025))
                ci_high = float(np.quantile(bootstrap_deltas, 0.975))
                lower_tail = (float(np.sum(bootstrap_deltas <= 0.0)) + 1.0) / (len(bootstrap_deltas) + 1.0)
                upper_tail = (float(np.sum(bootstrap_deltas >= 0.0)) + 1.0) / (len(bootstrap_deltas) + 1.0)
                p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
            else:
                inference_method = "descriptive_scalar_only"
                paired_bootstrap_count = 0
                delta_mean = float(our["value"] - baseline["value"])
                delta_std = None; ci_low = None; ci_high = None; p_value = None
            comparison_id = f"{row['task_id']}__ctx{row['context_bp']}__{baseline['row']['model_id']}"
            comparisons.append({
                "task_id": row["task_id"], "context_bp": row["context_bp"],
                "primary_metric": row["primary_metric"], "ours_model": "CropGenomeFM_step50000",
                "baseline_model": baseline["row"]["model_id"],
                "ours_value": our["value"], "baseline_value": baseline["value"],
                "pairing_verified": len(common) == 5 or paired_bootstrap_count >= 100,
                "inference_method": inference_method,
                "paired_seed_count": len(common), "delta_ours_minus_baseline": delta_mean,
                "paired_bootstrap_count": paired_bootstrap_count,
                "delta_std": delta_std, "delta_ci95_low": ci_low, "delta_ci95_high": ci_high,
                "paired_p_value": p_value, "winner": "ours" if delta_mean > 0 else "baseline" if delta_mean < 0 else "tie",
                "comparison_id": comparison_id,
            })
    _holm_adjust(comparisons)
    trajectory = [{
        "task_id": item["row"]["task_id"], "context_bp": item["row"]["context_bp"],
        "checkpoint_step": item["row"]["checkpoint_step"],
        "primary_metric": item["row"]["primary_metric"], "primary_value": item["value"],
        "evidence_class": "checkpoint_trajectory_monitoring", "run_key": item["row"]["run_key"],
    } for item in successful.values() if item["row"]["model_id"] == "CropGenomeFM"]
    trajectory.sort(key=lambda item: (item["task_id"], item["context_bp"], item["checkpoint_step"]))
    limitation_rows = [{
        "scope": "A04", "code": "within_species_official_split",
        "statement": "species groups cross splits by design; exact sequence and source-record overlap are audited separately",
        "severity": "interpretation_boundary", "limitation_id": "A04_scope",
    }, {
        "scope": "all_license_overrides", "code": "license_unverified_user_override",
        "statement": "internal computation authorization is not a redistribution license",
        "severity": "distribution_boundary", "limitation_id": "license_override",
    }]
    if low_homology_equals_full_test:
        limitation_rows.append({
            "scope": "B13-B17", "code": "low_homology_cohort_equals_full_test",
            "statement": "the 80% identity/80% query-coverage sensitivity cohort equals the full test set and adds no subset discrimination",
            "severity": "sensitivity_not_informative", "limitation_id": "low_homology_full_test",
        })
    output = final_root / "final_report"; output.mkdir(parents=True, exist_ok=True)
    result_path = output / "row_results.tsv"
    seed_path = output / "seed_source_data.tsv"
    bootstrap_path = output / "bootstrap_source_data.tsv"
    comparison_path = output / "paired_comparisons.tsv"
    trajectory_path = output / "checkpoint_trajectory.tsv"
    status_path = output / "matrix_status.tsv"
    limitation_path = output / "limitations.tsv"
    _write_tsv(result_path, result_rows, [
        "task_id", "model_id", "checkpoint_step", "context_bp", "primary_metric",
        "primary_value", "seed_mean", "seed_std", "seed_ci95_low", "seed_ci95_high",
        "seed_count", "uncertainty_method", "ci95_low", "ci95_high",
        "bootstrap_replicates", "run_key",
    ])
    _write_tsv(seed_path, seed_rows, [
        "task_id", "model_id", "checkpoint_step", "context_bp", "primary_metric",
        "seed", "value", "run_key",
    ])
    _write_tsv(bootstrap_path, bootstrap_rows, [
        "task_id", "model_id", "checkpoint_step", "context_bp", "primary_metric",
        "replicate", "value", "resample_indices_sha256", "run_key",
    ])
    _write_tsv(comparison_path, comparisons, [
        "task_id", "context_bp", "primary_metric", "ours_model", "baseline_model",
        "ours_value", "baseline_value", "pairing_verified", "inference_method",
        "paired_seed_count", "paired_bootstrap_count", "delta_ours_minus_baseline",
        "delta_std", "delta_ci95_low", "delta_ci95_high", "paired_p_value",
        "holm_adjusted_p", "winner", "comparison_id",
    ])
    _write_tsv(trajectory_path, trajectory, [
        "task_id", "context_bp", "checkpoint_step", "primary_metric", "primary_value",
        "evidence_class", "run_key",
    ])
    _write_tsv(status_path, status_rows, [
        "cell_type", "task_id", "model_id", "context_bp", "checkpoint_step",
        "status", "reason", "cell_id",
    ])
    _write_tsv(limitation_path, limitation_rows, [
        "scope", "code", "statement", "severity", "limitation_id",
    ])
    summary = {
        "status": audit["status"], "report_protocol_id": REPORT_PROTOCOL_ID,
        "plan_sha256": plan_sha, "rows_total": len(plan["rows"]),
        "rows_complete": sum(row["status"] == "complete" for row in status_rows),
        "rows_terminal_failed": sum(row["status"] == "terminal_failed" for row in status_rows),
        "not_applicable_cells": sum(row["status"] == "not_applicable" for row in status_rows),
        "paired_comparisons": len(comparisons),
        "seed_source_rows": len(seed_rows), "bootstrap_source_rows": len(bootstrap_rows),
        "limitations": len(limitation_rows),
        "test_used_for_checkpoint_selection": False,
        "checkpoint_trajectory_evidence_class": "monitoring/development",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    summary_path = output / "summary.json"; _atomic_json(summary_path, summary)
    audit_path = final_root / "FINAL_CLOSURE_AUDIT.json"
    artifacts = [
        {"path": str(path), "sha256": sha256_path(path), "size_bytes": path.stat().st_size}
        for path in (
            audit_path, result_path, seed_path, bootstrap_path, comparison_path,
            trajectory_path, status_path, limitation_path, summary_path,
        )
    ]
    receipt = {
        "status": "ok", "report_protocol_id": REPORT_PROTOCOL_ID,
        "plan_sha256": plan_sha, "closure_status": audit["status"],
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    _atomic_json(output / "FINAL_REPORT_RECEIPT.json", receipt)
    return {**summary, "receipt": str(output / "FINAL_REPORT_RECEIPT.json")}
