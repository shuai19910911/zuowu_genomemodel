"""Immutable one-checkpoint downstream snapshot requested during live training."""

import hashlib
import json
import math
from pathlib import Path

from .final_protocol import (
    EXCLUDED_TASK_IDS, _contexts, _supports_task, build_execution_plan,
    checkpoint_steps,
)


SNAPSHOT_PROTOCOL_ID = "cropgenome_downstream_latest_checkpoint_snapshot_v1"
CHECKPOINT_SET_PROTOCOL_ID = "cropgenome_downstream_explicit_checkpoint_set_v1"


def derive_memory_budgets(required_keys, measured_peak_mib,
                          safety_factor=1.25, fixed_margin_mib=256,
                          minimum_budget_mib=1536, maximum_budget_mib=10240):
    budgets = {}
    quantum = 256
    for key in required_keys:
        model_id = str(key).split(":", 1)[0]
        peak = measured_peak_mib.get(model_id)
        if peak is None or int(peak) <= 0:
            budget = int(maximum_budget_mib)
        else:
            raw = float(peak) * float(safety_factor) + int(fixed_margin_mib)
            budget = int(math.ceil(raw / quantum) * quantum)
            budget = max(int(minimum_budget_mib), min(int(maximum_budget_mib), budget))
        budgets[str(key)] = budget
    return budgets


def _mode_for_task(task):
    if task["task_id"] == "B13":
        return "token"
    if str(task["task_kind"]).startswith("zero_shot"):
        return "zero_shot"
    return "pooled"


def required_memory_budget_keys(registry, checkpoint_step,
                                excluded_task_ids=EXCLUDED_TASK_IDS):
    del checkpoint_step
    excluded_task_ids = set(map(str, excluded_task_ids or ()))
    keys = set()
    for task in registry["tasks"]:
        if task["task_id"] in excluded_task_ids or task["task_id"] == "B17":
            continue
        for model in registry["models"]:
            if model["kind"] == "simple_baseline" or not _supports_task(model, task):
                continue
            for context in _contexts(model, task):
                keys.add(f"{model['model_id']}:{int(context)}:{_mode_for_task(task)}")
    return sorted(keys)


def production_profile_contexts(registry):
    contexts = {}
    for key in required_memory_budget_keys(registry, checkpoint_step=0):
        model_id, raw_context, _mode = key.rsplit(":", 2)
        contexts[model_id] = max(contexts.get(model_id, 0), int(raw_context))
    return dict(sorted(contexts.items()))


def _checkpoint_set_rows(base_plan, selected_steps):
    template_step = checkpoint_steps()[0]
    steps = sorted({int(step) for step in selected_steps})
    rows = [
        dict(source) for source in base_plan["rows"]
        if source["checkpoint_scope"] == "shared"
    ]
    templates = [
        source for source in base_plan["rows"]
        if source["model_id"] == "CropGenomeFM"
        and source["checkpoint_scope"] == "step"
        and int(source["checkpoint_step"]) == int(template_step)
    ]
    old_scope = f"step{int(template_step):05d}"
    for step in steps:
        new_scope = f"step{int(step):05d}"
        for source in templates:
            row = dict(source)
            row["checkpoint_step"] = int(step)
            row["run_key"] = row["run_key"].replace(old_scope, new_scope, 1)
            rows.append(row)
    return sorted(rows, key=lambda row: row["run_key"])


def _snapshot_rows(base_plan, checkpoint_step):
    return _checkpoint_set_rows(base_plan, [checkpoint_step])


def _identity_steps(checkpoint_identities):
    steps = []
    for key, identity in sorted((checkpoint_identities or {}).items()):
        prefix = "step_"
        if not str(key).startswith(prefix):
            raise RuntimeError(f"invalid checkpoint identity key: {key}")
        try:
            step = int(str(key)[len(prefix):])
        except ValueError as error:
            raise RuntimeError(f"invalid checkpoint identity key: {key}") from error
        if not isinstance(identity, dict) or not identity.get("path") or not identity.get("sha256"):
            raise RuntimeError(f"incomplete checkpoint identity: {key}")
        steps.append(step)
    if not steps or len(steps) != len(set(steps)):
        raise RuntimeError("explicit checkpoint set must contain unique checkpoints")
    return sorted(steps)


def build_checkpoint_set_plan(registry, checkpoint_identities,
                              project_model_config, gpu_scheduler_policy,
                              excluded_task_ids=EXCLUDED_TASK_IDS):
    steps = _identity_steps(checkpoint_identities)
    if not isinstance(project_model_config, dict) or not (
        project_model_config.get("path") and project_model_config.get("sha256")
    ):
        raise RuntimeError("explicit checkpoint set requires a frozen project model config")
    base = build_execution_plan(
        registry, checkpoint_identities=dict(checkpoint_identities),
        checkpoint_identity_state="frozen_complete",
        excluded_task_ids=excluded_task_ids,
    )
    plan = dict(base)
    plan.pop("plan_sha256", None)
    plan["protocol_id"] = CHECKPOINT_SET_PROTOCOL_ID
    plan["checkpoint_schedule"] = {
        "selection": "explicit_manual_checkpoint_set", "steps": steps,
    }
    plan["checkpoint_identity_state"] = "frozen_complete"
    plan["checkpoint_identities"] = dict(checkpoint_identities)
    plan["project_model_config"] = dict(project_model_config)
    plan["rows"] = _checkpoint_set_rows(base, steps)
    policy = dict(gpu_scheduler_policy or {})
    if policy.get("mode") != "memory_packed":
        raise RuntimeError("explicit checkpoint set requires memory_packed GPU policy")
    max_tasks_per_gpu = int(policy.get("max_tasks_per_gpu", 1))
    if not 1 <= max_tasks_per_gpu <= 3:
        raise RuntimeError("max_tasks_per_gpu must be between 1 and 3")
    policy.setdefault("foreign_compute_allowed", False)
    policy.setdefault("unknown_compute_blocks", True)
    policy.setdefault("do_not_signal_foreign_processes", True)
    if (
        policy["unknown_compute_blocks"] is not True
        or policy["do_not_signal_foreign_processes"] is not True
    ):
        raise RuntimeError("unknown GPU occupancy must remain fail-closed")
    policy.update({
        "max_tasks_per_gpu": max_tasks_per_gpu,
        "nvitop_is_benign": True,
        "required_hostname": "gpu05",
        "allowed_gpu_indices": list(range(7)),
    })
    required = set(required_memory_budget_keys(
        registry, checkpoint_step=0, excluded_task_ids=excluded_task_ids,
    ))
    missing = sorted(required - set((policy.get("budgets_mib") or {}).keys()))
    if missing:
        raise RuntimeError("missing GPU memory budgets: " + ", ".join(missing))
    plan["gpu_scheduler_policy"] = policy
    plan["test_policy"] = {
        **base["test_policy"],
        "checkpoint_selection_rule": "explicit_manual_checkpoint_set",
        "checkpoint_trajectory_is_monitoring_evidence": True,
        "snapshot_is_final_model_selection": False,
        "final_checkpoint_is_predeclared_step_50000": False,
        "metrics_sealed_until_matrix_complete": True,
    }
    implementation = Path(__file__).resolve()
    plan["snapshot_implementation"] = {
        "path": str(implementation),
        "sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        "size_bytes": implementation.stat().st_size,
    }
    canonical = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    plan["plan_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def build_latest_snapshot_plan(registry, checkpoint_step, checkpoint_identity,
                               gpu_scheduler_policy):
    step = int(checkpoint_step)
    identity_key = f"step_{step:08d}"
    base = build_execution_plan(
        registry,
        checkpoint_identities={identity_key: dict(checkpoint_identity)},
        checkpoint_identity_state="frozen_complete",
    )
    plan = dict(base)
    plan.pop("plan_sha256", None)
    plan["protocol_id"] = SNAPSHOT_PROTOCOL_ID
    plan["checkpoint_schedule"] = {
        "selection": "chronologically_latest_complete_checkpoint_at_freeze",
        "steps": [step],
    }
    plan["checkpoint_identity_state"] = "frozen_complete"
    plan["checkpoint_identities"] = {identity_key: dict(checkpoint_identity)}
    plan["rows"] = _snapshot_rows(base, step)
    policy = dict(gpu_scheduler_policy or {})
    if policy.get("mode") != "memory_packed":
        raise RuntimeError("latest checkpoint snapshot requires memory_packed GPU policy")
    max_tasks_per_gpu = int(policy.get("max_tasks_per_gpu", 1))
    if not 1 <= max_tasks_per_gpu <= 3:
        raise RuntimeError("max_tasks_per_gpu must be between 1 and 3")
    policy.setdefault("foreign_compute_allowed", False)
    policy.setdefault("unknown_compute_blocks", True)
    policy.setdefault("do_not_signal_foreign_processes", True)
    if (
        policy["unknown_compute_blocks"] is not True
        or policy["do_not_signal_foreign_processes"] is not True
    ):
        raise RuntimeError("unknown GPU occupancy must remain fail-closed")
    policy.update({
        "max_tasks_per_gpu": max_tasks_per_gpu,
        "nvitop_is_benign": True,
        "required_hostname": "gpu05",
        "allowed_gpu_indices": list(range(7)),
    })
    required = set(required_memory_budget_keys(registry, step))
    provided = set((policy.get("budgets_mib") or {}).keys())
    missing = sorted(required - provided)
    if missing:
        raise RuntimeError("missing GPU memory budgets: " + ", ".join(missing))
    plan["gpu_scheduler_policy"] = policy
    plan["test_policy"] = {
        **base["test_policy"],
        "checkpoint_selection_rule": "chronologically_latest_complete_checkpoint",
        "snapshot_is_final_model_selection": False,
        "final_checkpoint_is_predeclared_step_50000": False,
        "metrics_sealed_until_matrix_complete": True,
    }
    implementation = Path(__file__).resolve()
    plan["snapshot_implementation"] = {
        "path": str(implementation),
        "sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        "size_bytes": implementation.stat().st_size,
    }
    canonical = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    plan["plan_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan
