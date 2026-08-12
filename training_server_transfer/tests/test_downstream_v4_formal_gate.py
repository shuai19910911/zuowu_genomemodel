import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.formal_gate import (
    _safe_gpu_scheduler_policy,
    _valid_migrated_model_smoke,
    authorize_formal_execution,
    valid_formal_execution_authorization,
    write_formal_execution_authorization_receipt,
)


def test_formal_gate_accepts_only_bounded_budgeted_gpu_colocation():
    policy = {
        "mode": "memory_packed", "max_workers": 3,
        "max_tasks_per_gpu": 3, "foreign_compute_allowed": True,
        "unknown_compute_blocks": True,
        "do_not_signal_foreign_processes": True,
        "reserved_headroom_mib": 768, "minimum_runtime_headroom_mib": 768,
        "nvitop_is_benign": True, "required_hostname": "gpu05",
        "allowed_gpu_indices": list(range(7)),
    }
    assert _safe_gpu_scheduler_policy({"gpu_scheduler_policy": policy}) is True
    for update in (
        {"max_tasks_per_gpu": 4},
        {"max_workers": 4},
        {"unknown_compute_blocks": False},
        {"do_not_signal_foreign_processes": False},
    ):
        assert _safe_gpu_scheduler_policy({
            "gpu_scheduler_policy": {**policy, **update},
        }) is False


def test_formal_authorization_receipt_binds_plan_evidence_and_checkpoint_stat(tmp_path):
    final_root = tmp_path / "final"
    final_root.mkdir()
    evidence = final_root / "DATASET_INTEGRITY_AUDIT.json"
    evidence.write_text('{"status":"ok"}\n')
    checkpoint = tmp_path / "step.pt"
    checkpoint.write_bytes(b"checkpoint")
    stat = checkpoint.stat()
    plan = {
        "plan_sha256": "frozen-plan",
        "checkpoint_identity_state": "frozen_complete",
        "model_ids": ["CropGenomeFM"],
        "comparison_policy": {
            "license_execution_policy": "user_authorized_direct_use",
            "license_metadata_is_execution_gate": False,
            "internal_pretraining_ablations": "disabled_by_user",
        },
        "checkpoint_identities": {
            "step_00050000": {
                "path": str(checkpoint.resolve()),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            },
        },
    }
    receipt = write_formal_execution_authorization_receipt(
        final_root, plan, [evidence], {"dataset_records": 53, "model_smokes": 14},
    )
    assert receipt["status"] == "authorized"
    assert valid_formal_execution_authorization(final_root, "frozen-plan") is not None
    evidence.write_text('{"status":"changed"}\n')
    assert valid_formal_execution_authorization(final_root, "frozen-plan") is None


def test_formal_authorization_receipt_rejects_draft_plan(tmp_path):
    with pytest.raises(RuntimeError, match="frozen_complete"):
        write_formal_execution_authorization_receipt(
            tmp_path, {"plan_sha256": "draft", "checkpoint_identity_state": "draft_incomplete"},
            [], {},
        )


def test_formal_authorization_receipt_rejects_unknown_compute_cotenancy(tmp_path):
    plan = {
        "plan_sha256": "unsafe", "checkpoint_identity_state": "frozen_complete",
        "model_ids": ["CropGenomeFM"],
        "comparison_policy": {
            "license_execution_policy": "user_authorized_direct_use",
            "license_metadata_is_execution_gate": False,
            "internal_pretraining_ablations": "disabled_by_user",
        },
        "checkpoint_identities": {},
        "gpu_scheduler_policy": {
            "mode": "memory_packed", "max_tasks_per_gpu": 3,
            "foreign_compute_allowed": True, "unknown_compute_blocks": False,
            "do_not_signal_foreign_processes": True,
            "reserved_headroom_mib": 1024, "minimum_runtime_headroom_mib": 512,
            "nvitop_is_benign": True, "required_hostname": "gpu05",
            "allowed_gpu_indices": list(range(7)),
        },
    }
    with pytest.raises(RuntimeError, match="regressed scientific policy"):
        write_formal_execution_authorization_receipt(tmp_path, plan, [], {})


def test_high_level_formal_authorization_writes_only_validated_evidence(monkeypatch, tmp_path):
    evidence = tmp_path / "gate.json"
    evidence.write_text('{"status":"ok"}\n')
    checkpoint = tmp_path / "step.pt"
    checkpoint.write_bytes(b"checkpoint")
    stat = checkpoint.stat()
    plan = {
        "plan_sha256": "frozen-plan", "checkpoint_identity_state": "frozen_complete",
        "model_ids": ["CropGenomeFM"],
        "comparison_policy": {
            "license_execution_policy": "user_authorized_direct_use",
            "license_metadata_is_execution_gate": False,
            "internal_pretraining_ablations": "disabled_by_user",
        },
        "checkpoint_identities": {"step": {
            "path": str(checkpoint), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }},
    }
    monkeypatch.setattr(
        "downstream_v4.formal_gate.validate_formal_prerequisites",
        lambda *args: ([evidence], {"dataset_records": 53, "model_smokes": 14}),
    )
    receipt = authorize_formal_execution(plan, {}, tmp_path, tmp_path)
    assert receipt["status"] == "authorized"
    assert valid_formal_execution_authorization(tmp_path, "frozen-plan") is not None


def test_migrated_smoke_requires_exact_contract_receipt_and_legacy_validation(tmp_path):
    final_root = tmp_path / "final"
    receipt = final_root / "model_smokes" / "model-0" / "SMOKE_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"status":"ok"}\n')
    records = []
    for index in range(14):
        model_id = f"model-{index}"
        model_receipt = final_root / "model_smokes" / model_id / "SMOKE_RECEIPT.json"
        if index:
            model_receipt.parent.mkdir(parents=True)
            model_receipt.write_text('{"status":"ok"}\n')
        raw = model_receipt.read_bytes()
        records.append({
            "model_id": model_id,
            "legacy_contract_sha256": f"legacy-{index}",
            "current_contract_sha256": f"current-{index}",
            "legacy_receipt_path": str(model_receipt.resolve()),
            "legacy_receipt_size_bytes": len(raw),
            "legacy_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        })
    migration = final_root / "controller" / "MEMORY_OPTIMIZATION_SMOKE_MIGRATION.json"
    migration.parent.mkdir(parents=True)
    migration.write_text(json.dumps({
        "status": "ok",
        "protocol_id": "memory_optimization_smoke_contract_migration_v1",
        "models_total": 14,
        "records": records,
    }) + "\n")
    amendment = {"supporting_evidence": [{"path": str(migration.resolve())}]}
    validator = lambda root, model, contract: (
        {"status": "ok"} if (model, contract) == ("model-0", "legacy-0") else None
    )
    assert _valid_migrated_model_smoke(
        amendment, final_root, "model-0", "current-0", validator,
    )
    receipt.write_text('{"status":"changed"}\n')
    assert not _valid_migrated_model_smoke(
        amendment, final_root, "model-0", "current-0", validator,
    )
