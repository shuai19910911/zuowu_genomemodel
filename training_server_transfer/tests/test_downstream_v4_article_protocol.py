import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.article_protocol import (
    ARTICLE_GUIDED_PROTOCOL_ID,
    build_article_guided_checkpoint_plan,
    load_article_profile,
)
from downstream_v4.formal_gate import _rebuild_live_plan
from downstream_v4.formal_gate import _expected_dataset_task_ids


def test_article_guided_checkpoint_plan_is_plant_only_and_single_checkpoint():
    registry = json.loads(
        (ROOT / "training_server_transfer/configs/cropgenome_downstream_v4.json").read_text()
    )
    profile = load_article_profile(
        ROOT / "training_server_transfer/configs/downstream_article_guided_v1.json"
    )
    model_config = ROOT / "training_server_transfer/configs/model_large.json"
    scheduler = json.loads(
        (ROOT / "training_server_transfer/configs/downstream_article_gpu05_2080ti.json").read_text()
    )["policy"]
    identity = {
        "step_00040000": {
            "path": "/tmp/checkpoint_step_00040000.pt",
            "sha256": "1" * 64,
            "size_bytes": 123,
            "checkpoint_step": 40000,
            "protocol_id": "cropgenome_checkpoint_identity_v1",
        }
    }
    plan = build_article_guided_checkpoint_plan(
        registry,
        checkpoint_identities=identity,
        project_model_config={
            "path": str(model_config),
            "sha256": "2" * 64,
            "size_bytes": model_config.stat().st_size,
        },
        gpu_scheduler_policy=scheduler,
        profile=profile,
    )

    expected = set(profile["core_task_ids"] + profile["supplement_task_ids"])
    observed = {row["task_id"] for row in plan["rows"]}
    assert plan["protocol_id"] == ARTICLE_GUIDED_PROTOCOL_ID
    assert plan["checkpoint_schedule"]["steps"] == [40000]
    assert plan["task_selection"]["task_ids"] == sorted(expected)
    assert observed == expected
    assert {"B11", "B12"} <= observed
    assert all(row["seeds"] == [13, 17, 42, 123, 997] for row in plan["rows"])
    assert all(
        next(task for task in registry["tasks"] if task["task_id"] == task_id)["organism_scope"]
        == "plant_or_crop"
        for task_id in observed
    )
    assert plan["article_guided_evaluation"]["complete_task_count"] == 25
    assert plan["test_policy"]["checkpoint_selection_rule"] == "explicit_user_supplied_checkpoint"
    assert plan["gpu_scheduler_policy"]["max_workers"] == 3
    assert plan["gpu_scheduler_policy"]["max_tasks_per_gpu"] == 3
    assert plan["gpu_scheduler_policy"]["foreign_compute_allowed"] is True
    assert plan["gpu_scheduler_policy"]["unknown_compute_blocks"] is True
    assert plan["gpu_scheduler_policy"]["do_not_signal_foreign_processes"] is True
    assert max(scheduler["budgets_mib"].values()) + scheduler["reserved_headroom_mib"] <= 11264
    a01 = next(row for row in plan["rows"] if row["task_id"] == "A01")
    d01 = next(row for row in plan["rows"] if row["task_id"] == "D01")
    assert a01["few_shot_regimes"] == [1, 10]
    assert a01["training_row_cap"] == 100000
    assert d01["few_shot_regimes"] == []
    assert d01["training_row_cap"] is None
    rebuilt = _rebuild_live_plan(plan, registry)
    assert rebuilt["plan_sha256"] == plan["plan_sha256"]
    expected_datasets = _expected_dataset_task_ids(plan, registry)
    assert expected_datasets == expected - {"B17"}
