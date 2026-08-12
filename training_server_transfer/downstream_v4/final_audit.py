"""Fail-closed audit for the single final downstream release."""

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .final_controller import (
    _valid_cpu_row_receipt, _valid_group_receipt,
    _valid_terminal_cpu_failure, _valid_terminal_group_failure, build_gpu_groups,
)
from .data import sha256_path
from .dataset_audit import valid_dataset_audit
from .environment import valid_environment_receipt
from .final_protocol import build_execution_plan
from .formal_gate import (
    _implementation_drift_evidence,
    _rebuild_live_plan as _rebuild_protocol_live_plan,
    _valid_migrated_model_smoke,
)
from .latest_snapshot import build_checkpoint_set_plan
from .operational_amendment import valid_memory_optimization_amendment
from .model_smoke import (
    smoke_contract_sha256, valid_smoke_receipt, valid_terminal_smoke_failure,
)
from .registry import (
    FORBIDDEN_INTERNAL_MODEL_IDS, INTERNAL_PRETRAINING_ABLATIONS,
    LICENSE_EXECUTION_POLICY,
)
from .sensitivity import valid_sensitivity_receipt


def _write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _quick_receipt(path, plan_sha256=None):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return payload.get("status") == "ok" and (
        plan_sha256 is None or payload.get("plan_sha256") == plan_sha256
    )


def _quick_terminal(path, plan_sha256, identity_key, identity):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    canonical = dict(payload); stored = canonical.pop("receipt_sha256", None)
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return bool(
        payload.get("status") == "terminal_failed"
        and payload.get("plan_sha256") == plan_sha256
        and payload.get(identity_key) == identity
        and stored == digest
    )


def _cpu_row(row):
    kind = row["task_kind"]
    return row["execution_kind"] == "evaluation" and kind != "token_multiclass" and not kind.startswith("zero_shot")


def _checkpoint_inventory_failures(paths, final_root, deep=False):
    inventory_path = Path(final_root) / "CHECKPOINT_INVENTORY.json"
    try:
        records = json.loads(inventory_path.read_text(encoding="utf-8")).get("records", {})
    except (OSError, ValueError, AttributeError):
        records = {}
    failures = []
    for checkpoint in paths:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            continue
        record = records.get(str(checkpoint)); stat = checkpoint.stat()
        if not record:
            failures.append({"path": str(checkpoint), "reason": "missing_inventory_record"})
        elif int(record.get("size_bytes", -1)) != stat.st_size:
            failures.append({"path": str(checkpoint), "reason": "size_changed"})
        elif int(record.get("mtime_ns", -1)) != stat.st_mtime_ns:
            failures.append({"path": str(checkpoint), "reason": "mtime_changed"})
        elif deep and sha256_path(checkpoint) != record.get("sha256"):
            failures.append({"path": str(checkpoint), "reason": "sha256_changed"})
    return failures


def _rebuild_live_plan(plan, registry):
    return _rebuild_protocol_live_plan(
        plan, registry, build_execution_plan, build_checkpoint_set_plan,
    )


def audit_final_closure(plan, registry, project_root, final_root, deep=False):
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    plan_sha = plan["plan_sha256"]
    live_plan = _rebuild_live_plan(plan, registry)
    memory_amendment = valid_memory_optimization_amendment(
        final_root, plan, live_plan, project_root,
    ) if live_plan.get("plan_sha256") != plan_sha else None
    plan_matches_live_implementation = _implementation_drift_evidence(
        plan, live_plan, project_root, final_root,
    ) is not None
    groups = build_gpu_groups(plan)
    group_by_row = {
        run_key: group for group in groups for run_key in group["run_keys"]
    }
    group_ok = {}; group_terminal = {}
    for group in groups:
        receipt = final_root / "gpu_groups" / group["group_key"] / "GROUP_RECEIPT.json"
        failure = final_root / "gpu_groups" / group["group_key"] / "FAILED.json"
        group_ok[group["group_key"]] = bool(
            _valid_group_receipt(group, final_root, plan_sha)
            if deep else _quick_receipt(receipt, plan_sha)
        )
        group_terminal[group["group_key"]] = bool(
            _valid_terminal_group_failure(group, final_root, plan_sha)
            if deep else _quick_terminal(
                failure, plan_sha, "group_key", group["group_key"],
            )
        )
    row_states = Counter(); unresolved_rows = []; terminal_rows = []
    for row in plan["rows"]:
        key = row["run_key"]
        terminal_cpu = bool(
            _valid_terminal_cpu_failure(row, final_root, plan_sha)
            if deep else _quick_terminal(
                final_root / "results" / key / "TERMINAL_FAILURE.json",
                plan_sha, "run_key", key,
            )
        )
        if row["execution_kind"] == "analysis":
            ready = bool(
                valid_sensitivity_receipt(row, final_root, plan_sha)
                if deep else _quick_receipt(
                    final_root / "results" / key / "ROW_RECEIPT.json", plan_sha,
                )
            )
            state = "complete" if ready else "terminal_failed" if terminal_cpu else "missing_analysis"
        elif _cpu_row(row):
            cpu_ok = bool(
                _valid_cpu_row_receipt(row, final_root, plan_sha)
                if deep else _quick_receipt(final_root / "results" / key / "ROW_RECEIPT.json", plan_sha)
            )
            if row["model_kind"] == "simple_baseline":
                state = "complete" if cpu_ok else "terminal_failed" if terminal_cpu else "missing_cpu"
            else:
                group = group_by_row[key]
                group_key = group["group_key"]
                if group_terminal[group_key]:
                    state = "terminal_failed"
                elif not group_ok[group_key]:
                    state = "missing_gpu"
                elif cpu_ok:
                    state = "complete"
                elif terminal_cpu:
                    state = "terminal_failed"
                else:
                    state = "missing_cpu"
        else:
            group = group_by_row.get(key)
            if group and group_ok[group["group_key"]]:
                state = "complete"
            elif group and group_terminal[group["group_key"]]:
                state = "terminal_failed"
            else:
                state = "missing_gpu"
        row_states[state] += 1
        if state == "terminal_failed" and len(terminal_rows) < 500:
            terminal_rows.append({"run_key": key, "state": state})
        elif state != "complete" and len(unresolved_rows) < 200:
            unresolved_rows.append({"run_key": key, "state": state})
    dataset_index = final_root / "DATASET_INDEX.json"
    try:
        dataset_status = json.loads(dataset_index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        dataset_status = {}
    public_models = {
        row["model_id"]: row for row in registry["models"] if row["kind"] == "public_weight"
    }
    smoke_ok = []; smoke_terminal = []; smoke_missing = []
    for model_id, model in public_models.items():
        contract = smoke_contract_sha256(model, project_root)
        if (
            valid_smoke_receipt(final_root, model_id, contract)
            or _valid_migrated_model_smoke(
                memory_amendment, final_root, model_id, contract,
                valid_smoke_receipt,
            )
        ):
            smoke_ok.append(model_id)
        elif valid_terminal_smoke_failure(final_root, model_id, contract):
            smoke_terminal.append(model_id)
        else:
            smoke_missing.append(model_id)
    expected_steps = plan["checkpoint_schedule"]["steps"]
    checkpoint_root = project_root / "training_server_transfer/runs/Stage_B_continuation_3gpu_no_replacement_from_step14000/checkpoints"
    missing_checkpoints = [
        step for step in expected_steps if not (checkpoint_root / f"step_{step:08d}.pt").is_file()
    ]
    expected_checkpoint_paths = [
        checkpoint_root / f"step_{step:08d}.pt" for step in expected_steps
    ]
    checkpoint_inventory_failures = _checkpoint_inventory_failures(
        expected_checkpoint_paths, final_root, deep=deep,
    )
    resolved_rows = row_states["complete"] + row_states["terminal_failed"]
    blockers = []
    comparison = plan.get("comparison_policy") or {}
    if comparison.get("license_execution_policy") != LICENSE_EXECUTION_POLICY:
        blockers.append("license_execution_policy")
    if comparison.get("license_metadata_is_execution_gate") is not False:
        blockers.append("license_metadata_execution_gate")
    if comparison.get("internal_pretraining_ablations") != INTERNAL_PRETRAINING_ABLATIONS:
        blockers.append("internal_pretraining_ablation_policy")
    if set(plan.get("model_ids") or []) & FORBIDDEN_INTERNAL_MODEL_IDS:
        blockers.append("forbidden_internal_models")
    if any(
        row.get("model_kind") == "internal_control"
        or row.get("model_id") in FORBIDDEN_INTERNAL_MODEL_IDS
        for row in plan.get("rows") or []
    ):
        blockers.append("forbidden_internal_model_rows")
    if any(
        model.get("kind") == "internal_control"
        or model.get("model_id") in FORBIDDEN_INTERNAL_MODEL_IDS
        for model in registry.get("models") or []
    ):
        blockers.append("forbidden_internal_models_in_registry")
    if (final_root / "LICENSE_OVERRIDE_RECEIPT.json").exists():
        blockers.append("retired_license_override_receipt_active")
    if (final_root / "no_region").exists():
        blockers.append("retired_no_region_namespace_active")
    if not plan_matches_live_implementation: blockers.append("plan_implementation_drift")
    if plan.get("checkpoint_identity_state") != "frozen_complete": blockers.append("checkpoint_identities")
    environment_receipt_valid = bool(valid_environment_receipt(
        final_root / "ENVIRONMENT_RECEIPT.json",
        project_root / "training_server_transfer/configs/downstream_v4_environment.lock.json",
        plan_sha,
    ))
    if not environment_receipt_valid: blockers.append("environment_receipt")
    if dataset_status.get("status") != "ready": blockers.append("datasets")
    dataset_integrity_valid = bool(valid_dataset_audit(registry, final_root, deep=deep))
    if not dataset_integrity_valid: blockers.append("dataset_integrity_audit")
    if smoke_missing: blockers.append("public_model_smokes")
    if missing_checkpoints: blockers.append("scheduled_checkpoints")
    if checkpoint_inventory_failures: blockers.append("checkpoint_inventory")
    if resolved_rows != len(plan["rows"]): blockers.append("execution_rows")
    final_status = "blocked" if blockers else (
        "closed_with_failures" if row_states["terminal_failed"] or smoke_terminal else "ok"
    )
    payload = {
        "status": final_status,
        "protocol_id": plan["protocol_id"], "plan_sha256": plan_sha,
        "plan_matches_live_implementation": plan_matches_live_implementation,
        "checkpoint_identity_state": plan.get("checkpoint_identity_state"),
        "environment_receipt_valid": environment_receipt_valid,
        "deep_hash_validation": bool(deep), "rows_total": len(plan["rows"]),
        "rows_resolved": resolved_rows, "row_states": dict(row_states),
        "unresolved_rows_preview": unresolved_rows,
        "terminal_rows_preview": terminal_rows,
        "gpu_groups_total": len(groups), "gpu_groups_complete": sum(group_ok.values()),
        "gpu_groups_terminal_failed": sum(group_terminal.values()),
        "datasets": {"status": dataset_status.get("status"), "ready": dataset_status.get("ready")},
        "dataset_integrity_audit_valid": dataset_integrity_valid,
        "public_model_smokes_complete": sorted(smoke_ok),
        "public_model_smokes_terminal_failed": sorted(smoke_terminal),
        "public_model_smokes_missing": smoke_missing,
        "checkpoint_steps_missing": missing_checkpoints,
        "checkpoint_inventory_failures": checkpoint_inventory_failures,
        "blockers": blockers,
        "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(final_root / "FINAL_CLOSURE_AUDIT.json", payload)
    return payload
