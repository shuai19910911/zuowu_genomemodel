import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.latest_snapshot import (
    build_checkpoint_set_plan, build_latest_snapshot_plan, derive_memory_budgets,
    production_profile_contexts, required_memory_budget_keys,
)


REGISTRY = json.loads(
    (ROOT / "training_server_transfer/configs/cropgenome_downstream_v4.json").read_text()
)


def _identity(step):
    return {
        "path": f"/checkpoints/step_{step:08d}.pt", "size_bytes": 123,
        "mtime_ns": 456, "sha256": "a" * 64,
    }


def test_latest_snapshot_has_one_checkpoint_and_all_shared_comparators():
    step = 45500
    keys = required_memory_budget_keys(REGISTRY, step)
    policy = {
        "mode": "memory_packed", "reserved_headroom_mib": 1024,
        "minimum_runtime_headroom_mib": 512, "max_tasks_per_gpu": 1,
        "budgets_mib": {key: 7000 for key in keys},
    }
    plan = build_latest_snapshot_plan(REGISTRY, step, _identity(step), policy)
    assert plan["protocol_id"] == "cropgenome_downstream_latest_checkpoint_snapshot_v1"
    assert plan["checkpoint_identity_state"] == "frozen_complete"
    assert plan["checkpoint_schedule"]["steps"] == [step]
    assert set(plan["checkpoint_identities"]) == {f"step_{step:08d}"}
    assert len(plan["rows"]) == 1302
    assert sum(row["checkpoint_scope"] == "shared" for row in plan["rows"]) == 1167
    step_rows = [row for row in plan["rows"] if row["checkpoint_scope"] == "step"]
    assert len(step_rows) == 135
    assert {row["checkpoint_step"] for row in step_rows} == {step}
    assert all(f"step{step:05d}" in row["run_key"] for row in step_rows)
    assert len({row["run_key"] for row in plan["rows"]}) == len(plan["rows"])
    assert plan["test_policy"]["checkpoint_selection_rule"] == "chronologically_latest_complete_checkpoint"
    assert plan["test_policy"]["snapshot_is_final_model_selection"] is False
    assert plan["gpu_scheduler_policy"]["foreign_compute_allowed"] is False
    assert plan["gpu_scheduler_policy"]["unknown_compute_blocks"] is True
    assert plan["gpu_scheduler_policy"]["max_tasks_per_gpu"] == 1
    assert plan["gpu_scheduler_policy"]["required_hostname"] == "gpu05"
    assert len(plan["plan_sha256"]) == 64


def test_latest_snapshot_refuses_missing_group_memory_budget():
    step = 45500
    with pytest.raises(RuntimeError, match="missing GPU memory budgets"):
        build_latest_snapshot_plan(
            REGISTRY, step, _identity(step), {
                "mode": "memory_packed", "budgets_mib": {},
            },
        )


def test_explicit_checkpoint_set_plan_contains_only_requested_steps():
    steps = [40000, 45000, 50000]
    identities = {f"step_{step:08d}": _identity(step) for step in steps}
    keys = required_memory_budget_keys(REGISTRY, 0)
    model_config = {
        "path": "/configs/model_large.json", "size_bytes": 42,
        "mtime_ns": 99, "sha256": "b" * 64,
    }
    plan = build_checkpoint_set_plan(
        REGISTRY, identities, model_config, {
            "mode": "memory_packed", "reserved_headroom_mib": 768,
            "minimum_runtime_headroom_mib": 768, "max_tasks_per_gpu": 3,
            "foreign_compute_allowed": True, "unknown_compute_blocks": True,
            "budgets_mib": {key: 7000 for key in keys},
        },
    )
    assert plan["protocol_id"] == "cropgenome_downstream_explicit_checkpoint_set_v1"
    assert plan["checkpoint_schedule"] == {
        "selection": "explicit_manual_checkpoint_set",
        "steps": steps,
    }
    assert plan["checkpoint_identities"] == identities
    assert plan["project_model_config"] == model_config
    assert len(plan["rows"]) == 1572
    step_rows = [row for row in plan["rows"] if row["checkpoint_scope"] == "step"]
    assert len(step_rows) == 405
    assert {row["checkpoint_step"] for row in step_rows} == set(steps)
    assert plan["test_policy"]["checkpoint_selection_rule"] == "explicit_manual_checkpoint_set"
    assert plan["test_policy"]["checkpoint_trajectory_is_monitoring_evidence"] is True
    assert plan["test_policy"]["snapshot_is_final_model_selection"] is False
    assert plan["gpu_scheduler_policy"]["reserved_headroom_mib"] == 768
    assert plan["gpu_scheduler_policy"]["max_tasks_per_gpu"] == 3
    assert plan["gpu_scheduler_policy"]["foreign_compute_allowed"] is True
    assert plan["gpu_scheduler_policy"]["unknown_compute_blocks"] is True


def test_checkpoint_set_rejects_more_than_three_tasks_per_gpu():
    identities = {"step_00040000": _identity(40000)}
    keys = required_memory_budget_keys(REGISTRY, 0)
    with pytest.raises(RuntimeError, match="between 1 and 3"):
        build_checkpoint_set_plan(
            REGISTRY, identities,
            {"path": "/configs/model.json", "sha256": "b" * 64},
            {
                "mode": "memory_packed", "max_tasks_per_gpu": 4,
                "foreign_compute_allowed": True,
                "unknown_compute_blocks": True,
                "budgets_mib": {key: 7000 for key in keys},
            },
        )


def test_memory_budgets_are_derived_from_measured_model_peaks():
    keys = [
        "small:512:pooled", "small:512:zero_shot",
        "large:8192:pooled", "unmeasured:512:pooled",
    ]
    budgets = derive_memory_budgets(
        keys, {"small": 2000, "large": 8000},
        safety_factor=1.25, fixed_margin_mib=256,
        minimum_budget_mib=1536, maximum_budget_mib=10240,
    )
    assert budgets["small:512:pooled"] == 2816
    assert budgets["small:512:zero_shot"] == 2816
    assert budgets["large:8192:pooled"] == 10240
    assert budgets["unmeasured:512:pooled"] == 10240


def test_production_profiles_cover_every_gpu_model_at_maximum_context():
    contexts = production_profile_contexts(REGISTRY)
    assert len(contexts) == 15
    assert contexts["CropGenomeFM"] == 8192
    assert "kmer_logistic" not in contexts
    assert all(value >= 512 for value in contexts.values())
