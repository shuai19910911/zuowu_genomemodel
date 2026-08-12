import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.final_protocol import (
    build_execution_plan, checkpoint_steps, write_execution_plan,
)


REGISTRY = json.loads(
    (ROOT / "training_server_transfer/configs/cropgenome_downstream_v4.json").read_text()
)


def rows(plan, task_id=None, model_id=None, checkpoint_step="any"):
    result = plan["rows"]
    if task_id is not None:
        result = [row for row in result if row["task_id"] == task_id]
    if model_id is not None:
        result = [row for row in result if row["model_id"] == model_id]
    if checkpoint_step != "any":
        result = [row for row in result if row.get("checkpoint_step") == checkpoint_step]
    return result


def test_final_checkpoint_schedule_is_frozen_from_16000_to_50000():
    assert checkpoint_steps() == list(range(16000, 50001, 2000))
    assert len(checkpoint_steps()) == 18


def test_plan_binds_runtime_implementation_and_checkpoint_identity_state(tmp_path):
    plan = build_execution_plan(REGISTRY)
    assert plan["checkpoint_identity_state"] == "draft_unbound"
    assert plan["checkpoint_identities"] == {}
    assert len(plan["model_runtime_semantic_sha256"]) == 64
    assert "training_server_transfer/downstream_v4/metrics.py" in plan["implementation_assets"]
    assert "training_server_transfer/downstream_v4/formal_gate.py" in plan["implementation_assets"]
    assert plan["comparison_policy"]["head_input_dim"] == 256
    assert all(row["primary_metric_direction"] == "higher_is_better" for row in plan["rows"])
    frozen = build_execution_plan(
        REGISTRY, checkpoint_identities={"checkpoint": {"sha256": "a" * 64}},
        checkpoint_identity_state="frozen_complete",
    )
    assert frozen["checkpoint_identity_state"] == "frozen_complete"
    assert frozen["plan_sha256"] != plan["plan_sha256"]

    output = tmp_path / "FINAL_PROTOCOL.json"
    written = write_execution_plan(REGISTRY, output)
    assert written["checkpoint_identity_state"] == "draft_incomplete"
    assert not (tmp_path / "LICENSE_OVERRIDE_RECEIPT.json").exists()


def test_final_plan_excludes_only_edta_tasks_and_includes_all_models():
    plan = build_execution_plan(REGISTRY)
    assert plan["profile"] == "full"
    assert plan["excluded_task_ids"] == ["B11", "B12"]
    assert len(plan["task_ids"]) == 54
    assert "B17" in plan["task_ids"]
    assert "CropGenomeFM_no_region" not in plan["model_ids"]
    assert "CropGenomeFM_no_region" not in {row["model_id"] for row in plan["rows"]}
    assert "CropGenomeFM_random_init" not in plan["model_ids"]
    assert not any(model["kind"] == "internal_control" for model in REGISTRY["models"])
    assert {row["model_id"] for row in plan["rows"]} == {
        model["model_id"] for model in REGISTRY["models"]
    }
    assert not rows(plan, "B11")
    assert not rows(plan, "B12")


def test_public_baselines_are_shared_but_cropgenome_is_checkpoint_specific():
    plan = build_execution_plan(REGISTRY)
    agro = rows(plan, "A01", "AgroNT_1B")
    assert {(row["context_bp"], row["checkpoint_scope"]) for row in agro} == {(512, "shared")}
    crop = rows(plan, "A01", "CropGenomeFM")
    assert {row["checkpoint_step"] for row in crop} == set(checkpoint_steps())
    assert {row["context_bp"] for row in crop} == {512, 2048, 8192}
    assert all(row["checkpoint_scope"] == "step" for row in crop)


def test_context_and_capability_tracks_are_honest():
    plan = build_execution_plan(REGISTRY)
    assert {row["context_bp"] for row in rows(plan, "A01", "PlantCAD2_Small")} == {512, 2048, 8192}
    assert not rows(plan, "C05", "AgroNT_1B")
    assert {row["context_bp"] for row in rows(plan, "C05", "PlantDNAMamba_BPE")} == {1000}
    assert {row["context_bp"] for row in rows(plan, "C28", "CropGenomeFM")} == {8192}
    assert not rows(plan, "C28", "GPN_Brassicales")
    assert rows(plan, "B13", "PlantCaduceus_l32")
    assert not rows(plan, "B13", "AgroNT_1B")
    assert rows(plan, "C17", "PlantCaduceus_l32")
    assert not rows(plan, "C17", "HyenaDNA_medium_160k")


def test_user_authorized_models_run_without_license_override_fields():
    plan = build_execution_plan(REGISTRY)
    assert plan["comparison_policy"]["license_execution_policy"] == "user_authorized_direct_use"
    assert "license_review_override" not in plan["comparison_policy"]
    for model_id in ("DNABERT2", "GENA_LM_BERT_base"):
        selected = rows(plan, "A01", model_id)
        assert selected
        assert all("license_override" not in row for row in selected)


def test_plan_keys_are_unique_and_test_policy_forbids_adaptive_selection():
    plan = build_execution_plan(REGISTRY)
    keys = [row["run_key"] for row in plan["rows"]]
    assert len(keys) == len(set(keys))
    assert len(plan["applicability"]) == len(plan["task_ids"]) * len(plan["model_ids"])
    assert any(item["status"] == "not_applicable" for item in plan["applicability"])
    assert plan["test_policy"]["metrics_sealed_until_matrix_complete"] is True
    assert plan["test_policy"]["checkpoint_selected_from_test"] is False
    assert plan["probe_protocols"]["pooled"] == "paired_seeded_train_bootstrap_hash256_linear_probe_v4"
    assert plan["probe_protocols"]["token"] == "seeded_hash256_token_linear_probe_v2"
    assert all(row["evaluation_protocol_id"] for row in plan["rows"])
    assert {row["cv_policy"] for row in plan["rows"] if row["task_id"] == "D01"} == {"leave_one_group_out"}
    assert {row["cv_policy"] for row in plan["rows"] if row["task_id"] == "A01"} == {"fixed_train_validation_test"}
    assert {row["cv_policy"] for row in plan["rows"] if row["task_id"] == "C28"} == {"official_test_only_zero_shot"}
    assert {row["evidence_scope"] for row in plan["rows"] if row["task_id"] == "C28"} == {"official_test_only_zero_shot_evidence"}


def test_active_protocol_contains_no_pretraining_ablation_assets():
    plan = build_execution_plan(REGISTRY)
    assert not any("no_region" in path for path in plan["implementation_assets"])
    assert not any(model.get("kind") == "internal_control" for model in REGISTRY["models"])
    assert plan["comparison_policy"]["internal_pretraining_ablations"] == "disabled_by_user"
