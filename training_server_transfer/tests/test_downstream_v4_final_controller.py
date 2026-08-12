import json
import fcntl
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.data import materialize_rows
from downstream_v4.final_controller import (
    _checkpoint_for_group, _checkpoint_record, _daemon_authorization_ready,
    _daemon_group_host_capacity,
    _daemon_launch_slots, _daemon_status,
    _group_lock_is_held,
    _group_namespace, _group_worker_command, _model_config_for_group,
    _host_memory_budget_for_group, _memory_policy_for_group,
    _resolve_daemon_worker_limit,
    _valid_terminal_cpu_failure, _valid_terminal_group_failure,
    _write_terminal_group_failure, build_gpu_groups, execute_cpu_row, execute_group,
    current_controller_status, group_readiness, record_cpu_row_failure,
)
from downstream_v4.final_protocol import build_execution_plan


REGISTRY = json.loads(
    (ROOT / "training_server_transfer/configs/cropgenome_downstream_v4.json").read_text()
)


def test_gpu_groups_partition_every_non_analysis_non_kmer_row_once():
    plan = build_execution_plan(REGISTRY)
    groups = build_gpu_groups(plan)
    observed = [key for group in groups for key in group["run_keys"]]
    expected = [
        row["run_key"] for row in plan["rows"]
        if row["execution_kind"] == "evaluation" and row["model_id"] != "kmer_logistic"
    ]
    assert sorted(observed) == sorted(expected)
    assert len(observed) == len(set(observed))


def test_gpu_group_never_mixes_scope_model_context_or_mode():
    groups = build_gpu_groups(build_execution_plan(REGISTRY))
    for group in groups:
        assert group["task_ids"] == sorted(set(group["task_ids"]))
        assert group["mode"] in {"pooled", "token", "zero_shot"}
        assert len(group["run_keys"]) == len(group["task_ids"])
    crop = next(
        group for group in groups
        if group["scope_key"] == "step16000" and group["model_id"] == "CropGenomeFM"
        and group["context_bp"] == 512 and group["mode"] == "pooled"
    )
    assert {"A01", "A02", "A03"} <= set(crop["task_ids"])
    assert "B13" not in crop["task_ids"]
    assert "C17" not in crop["task_ids"]
    token = next(
        group for group in groups
        if group["scope_key"] == "step16000" and group["model_id"] == "CropGenomeFM"
        and group["context_bp"] == 512 and group["mode"] == "token"
    )
    assert token["task_ids"] == ["B13"]
    zero = next(
        group for group in groups
        if group["scope_key"] == "step16000" and group["model_id"] == "CropGenomeFM"
        and group["context_bp"] == 512 and group["mode"] == "zero_shot"
    )
    assert "C17" in zero["task_ids"]


def test_group_keys_are_deterministic_and_unique():
    first = build_gpu_groups(build_execution_plan(REGISTRY))
    second = build_gpu_groups(build_execution_plan(REGISTRY))
    assert first == second
    assert len({group["group_key"] for group in first}) == len(first)


def test_group_lock_detects_worker_surviving_daemon_restart(tmp_path):
    group = {"group_key": "g"}
    lock_path = tmp_path / "gpu_groups/g/.run.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        assert _group_lock_is_held(group, tmp_path) is True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    assert _group_lock_is_held(group, tmp_path) is False
    absent = tmp_path / "gpu_groups/absent/.run.lock"
    assert _group_lock_is_held({"group_key": "absent"}, tmp_path) is False
    assert not absent.exists()


def test_live_status_does_not_repeat_stale_running_snapshot(monkeypatch, tmp_path):
    (tmp_path / "GPU_CONTROLLER_STATUS.json").write_text(json.dumps({
        "status": "running", "groups_external_active": ["g"],
        "updated_at": "2020-01-01T00:00:00+00:00",
    }))
    monkeypatch.setattr(
        "downstream_v4.final_controller.build_gpu_groups",
        lambda plan: [{"group_key": "g"}],
    )
    monkeypatch.setattr(
        "downstream_v4.final_controller._valid_group_receipt", lambda *args: None,
    )
    monkeypatch.setattr(
        "downstream_v4.final_controller._valid_terminal_group_failure", lambda *args: None,
    )
    monkeypatch.setattr(
        "downstream_v4.final_controller._group_lock_is_held", lambda *args: False,
    )
    status = current_controller_status(
        {"plan_sha256": "current", "checkpoint_identity_state": "draft_incomplete"},
        tmp_path,
    )
    assert status["status"] == "not_running"
    assert status["groups_active"] == []
    assert status["daemon_live"] is False
    assert status["stale_snapshot_detected"] is True


def test_checkpoint_hash_is_computed_once_and_reused(monkeypatch, tmp_path):
    checkpoint = tmp_path / "step.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    calls = []

    def counted(path):
        calls.append(Path(path))
        return "digest"

    monkeypatch.setattr("downstream_v4.final_controller.sha256_path", counted)
    first = _checkpoint_record(checkpoint, tmp_path)
    second = _checkpoint_record(checkpoint, tmp_path)
    assert first == second
    assert first["sha256"] == "digest"
    assert calls == [checkpoint]


def test_group_checkpoint_and_model_config_are_resolved_from_frozen_plan(tmp_path):
    project_root = tmp_path / "project"
    final_root = tmp_path / "final"
    checkpoint = tmp_path / "selected" / "stage_b_45000.pt"
    model_config = tmp_path / "configs" / "stage_b.json"
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b"checkpoint")
    model_config.parent.mkdir(); model_config.write_text("{}")
    final_root.mkdir()
    (final_root / "FINAL_PROTOCOL.json").write_text(json.dumps({
        "checkpoint_identities": {
            "step_00045000": {"path": str(checkpoint), "sha256": "a" * 64},
        },
        "project_model_config": {
            "path": str(model_config), "sha256": "b" * 64,
        },
    }))
    group = {"model_id": "CropGenomeFM", "checkpoint_step": 45000}
    assert _checkpoint_for_group(group, project_root, final_root) == checkpoint.resolve()
    assert _model_config_for_group(group, project_root, final_root) == model_config.resolve()


def test_group_readiness_refuses_frozen_checkpoint_identity_drift(tmp_path):
    checkpoint = tmp_path / "step_00045000.pt"
    checkpoint.write_bytes(b"current-checkpoint")
    config = tmp_path / "model.json"; config.write_text("{}")
    final_root = tmp_path / "final"; final_root.mkdir()
    (final_root / "FINAL_PROTOCOL.json").write_text(json.dumps({
        "checkpoint_identities": {"step_00045000": {
            "path": str(checkpoint), "size_bytes": checkpoint.stat().st_size + 1,
            "mtime_ns": checkpoint.stat().st_mtime_ns, "sha256": "a" * 64,
        }},
        "project_model_config": {
            "path": str(config), "size_bytes": config.stat().st_size,
            "mtime_ns": config.stat().st_mtime_ns, "sha256": "b" * 64,
        },
    }))
    group = {
        "model_id": "CropGenomeFM", "model_kind": "pretrained_checkpoint",
        "checkpoint_step": 45000, "task_ids": [],
    }
    ready, reason = group_readiness(group, {}, tmp_path, final_root)
    assert ready is False
    assert reason == "checkpoint_identity:step_00045000"


def test_group_parts_share_namespace_but_contexts_do_not():
    base = {
        "scope_key": "shared", "model_id": "model", "context_bp": 512,
        "mode": "pooled", "group_key": "shared__model__ctx512__pooled__part00",
    }
    other_part = {**base, "group_key": "shared__model__ctx512__pooled__part01"}
    other_context = {**base, "context_bp": 2048}
    assert _group_namespace(base) == _group_namespace(other_part)
    assert _group_namespace(base) != _group_namespace(other_context)


def test_group_rechecks_current_protocol_after_worker_start(monkeypatch, tmp_path):
    final_root = tmp_path / "final"
    final_root.mkdir()
    (final_root / "FINAL_PROTOCOL.json").write_text(json.dumps({
        "plan_sha256": "plan-at-worker-start",
        "checkpoint_identity_state": "draft_incomplete",
    }))
    executed = []

    def forbidden_execution(*args, **kwargs):
        executed.append(True)
        return {"status": "should-not-run"}

    monkeypatch.setattr(
        "downstream_v4.final_controller._execute_group_unlocked", forbidden_execution,
    )
    group = {
        "group_key": "shared__model__ctx512__pooled__part00",
        "scope_key": "shared", "model_id": "model", "context_bp": 512,
        "mode": "pooled",
    }
    with pytest.raises(RuntimeError, match="formal execution authorization revoked"):
        execute_group(group, {}, tmp_path, final_root, "plan-at-worker-start", tmp_path / "runner.py")
    assert executed == []


def test_group_requires_formal_gate_authorization_even_for_frozen_plan(monkeypatch, tmp_path):
    final_root = tmp_path / "final"
    final_root.mkdir()
    (final_root / "FINAL_PROTOCOL.json").write_text(json.dumps({
        "plan_sha256": "frozen-plan",
        "checkpoint_identity_state": "frozen_complete",
    }))
    executed = []
    monkeypatch.setattr(
        "downstream_v4.final_controller._execute_group_unlocked",
        lambda *args, **kwargs: executed.append(True),
    )
    group = {
        "group_key": "shared__model__ctx512__pooled__part00",
        "scope_key": "shared", "model_id": "model", "context_bp": 512,
        "mode": "pooled",
    }
    with pytest.raises(RuntimeError, match="formal execution authorization missing or stale"):
        execute_group(group, {}, tmp_path, final_root, "frozen-plan", tmp_path / "runner.py")
    assert executed == []


def test_one_terminal_group_failure_does_not_stop_unrelated_groups():
    keys = ["g0", "g1", "g2"]
    assert _daemon_status(keys, {"g0"}, {"g1"}) == "running"
    assert _daemon_status(keys, {"g0", "g2"}, {"g1"}) == "failed"
    assert _daemon_status(keys, set(keys), set()) == "ok"


def test_zero_max_workers_uses_frozen_worker_limit_and_cannot_relax_it():
    plan = {"gpu_scheduler_policy": {"max_workers": 5, "max_tasks_per_gpu": 1}}
    assert _resolve_daemon_worker_limit(0, plan, range(7)) == 5
    assert _resolve_daemon_worker_limit(3, plan, range(7)) == 3
    with pytest.raises(RuntimeError, match="exceeds frozen max_workers"):
        _resolve_daemon_worker_limit(6, plan, range(7))


def test_memory_unbounded_daemon_keeps_one_unclaimed_worker():
    assert _daemon_launch_slots(None, {"claimed"}, {"claimed"}) == 1
    assert _daemon_launch_slots(None, {"claimed", "waiting"}, {"claimed"}) == 0
    assert _daemon_launch_slots(None, set(), set()) == 1
    assert _daemon_launch_slots(3, {"a", "b"}, {"a", "b"}) == 1


def test_daemon_host_precheck_skips_heavy_group_but_allows_light_group():
    plan = {"gpu_scheduler_policy": {
        "mode": "memory_packed", "reserved_headroom_mib": 1024,
        "minimum_runtime_headroom_mib": 512, "max_tasks_per_gpu": 1,
        "foreign_compute_allowed": False, "unknown_compute_blocks": True,
        "nvitop_is_benign": True, "wait_timeout_seconds": 7200,
        "budgets_mib": {
            "CropGenomeFM:2048:zero_shot": 2304,
            "public:512:pooled": 1536,
        },
    }}
    claims = {"GPU-a": [{
        "host_memory_budget_mib": 49152, "child_pid": 123,
    }]}
    heavy = {"model_id": "CropGenomeFM", "context_bp": 2048, "mode": "zero_shot"}
    light = {"model_id": "public", "context_bp": 512, "mode": "pooled"}
    heavy_ready, heavy_report = _daemon_group_host_capacity(
        plan, heavy, claims, available_host_memory_mib=90000,
        process_rss_lookup=lambda pid: 10000,
    )
    light_ready, light_report = _daemon_group_host_capacity(
        plan, light, claims, available_host_memory_mib=90000,
        process_rss_lookup=lambda pid: 10000,
    )
    assert heavy_ready is False
    assert heavy_report["status"] == "blocked_host_memory_capacity"
    assert light_ready is True
    assert light_report["status"] == "ready"


def test_daemon_pauses_new_launches_when_formal_authorization_is_stale(
        monkeypatch, tmp_path):
    plan = {"plan_sha256": "plan"}
    monkeypatch.setattr(
        "downstream_v4.final_controller.valid_formal_execution_authorization",
        lambda *args: None,
    )
    assert _daemon_authorization_ready(plan, tmp_path) is False
    monkeypatch.setattr(
        "downstream_v4.final_controller.valid_formal_execution_authorization",
        lambda *args: {"status": "authorized"},
    )
    assert _daemon_authorization_ready(plan, tmp_path) is True


def test_daemon_honors_scoped_operational_authorization(monkeypatch, tmp_path):
    amendment = tmp_path / "OPERATIONAL_SCHEDULER_AMENDMENT.final_20260809T000000Z.json"
    amendment.write_text(json.dumps({
        "plan_sha256": "plan", "new_formal_worker_launch_allowed": False,
    }))
    monkeypatch.setattr(
        "downstream_v4.final_controller.valid_formal_execution_authorization",
        lambda *args: {"evidence": [{"path": str(amendment)}]},
    )
    assert _daemon_authorization_ready({"plan_sha256": "plan"}, tmp_path) is False
    amendment.write_text(json.dumps({
        "plan_sha256": "plan", "new_formal_worker_launch_allowed": True,
    }))
    assert _daemon_authorization_ready({"plan_sha256": "plan"}, tmp_path) is True


def test_group_worker_command_propagates_foreign_compute_policy(tmp_path):
    enabled = _group_worker_command(
        tmp_path / "runner.py", "group", tmp_path / "final", True, 0,
    )
    strict = _group_worker_command(
        tmp_path / "runner.py", "group", tmp_path / "final", False, 1,
    )
    assert enabled[-3:] == [
        "--allow-foreign-compute", "--max-tasks-per-gpu", "0",
    ]
    assert "--allow-foreign-compute" not in strict
    assert strict[-2:] == ["--max-tasks-per-gpu", "1"]


def test_memory_packed_group_policy_requires_exact_frozen_budget():
    plan = {"gpu_scheduler_policy": {
        "mode": "memory_packed", "reserved_headroom_mib": 1024,
        "minimum_runtime_headroom_mib": 512, "max_tasks_per_gpu": 1,
        "foreign_compute_allowed": False, "unknown_compute_blocks": True,
        "nvitop_is_benign": True, "wait_timeout_seconds": 7200,
        "budgets_mib": {"model:512:pooled": 3400},
    }}
    group = {"model_id": "model", "context_bp": 512, "mode": "pooled"}
    policy = _memory_policy_for_group(plan, group)
    assert policy["memory_budget_mib"] == 3400
    assert policy["reserved_headroom_mib"] == 1024
    assert policy["max_tasks_per_gpu"] == 1

    with pytest.raises(RuntimeError, match="missing frozen GPU memory budget"):
        _memory_policy_for_group(plan, {**group, "context_bp": 8192})

    unsafe = {"gpu_scheduler_policy": {
        **plan["gpu_scheduler_policy"], "foreign_compute_allowed": True,
    }}
    with pytest.raises(RuntimeError, match="invalid frozen GPU memory policy"):
        _memory_policy_for_group(unsafe, group)


def test_runtime_cannot_enable_foreign_compute_against_frozen_policy():
    plan = {"gpu_scheduler_policy": {
        "mode": "memory_packed", "reserved_headroom_mib": 1024,
        "minimum_runtime_headroom_mib": 512, "max_tasks_per_gpu": 1,
        "foreign_compute_allowed": False, "unknown_compute_blocks": True,
        "nvitop_is_benign": True, "wait_timeout_seconds": 7200,
        "budgets_mib": {"model:512:pooled": 3400},
    }}
    group = {"model_id": "model", "context_bp": 512, "mode": "pooled"}
    with pytest.raises(RuntimeError, match="relaxes frozen GPU memory policy"):
        _memory_policy_for_group(
            plan, group, foreign_compute_allowed=True,
        )


def test_runtime_cannot_remove_frozen_per_gpu_task_limit():
    plan = {"gpu_scheduler_policy": {
        "mode": "memory_packed", "reserved_headroom_mib": 1024,
        "minimum_runtime_headroom_mib": 512, "max_tasks_per_gpu": 1,
        "foreign_compute_allowed": False, "unknown_compute_blocks": True,
        "nvitop_is_benign": True, "wait_timeout_seconds": 7200,
        "budgets_mib": {"model:512:pooled": 3400},
    }}
    group = {"model_id": "model", "context_bp": 512, "mode": "pooled"}
    with pytest.raises(RuntimeError, match="relaxes frozen GPU memory policy"):
        _memory_policy_for_group(
            plan, group, max_tasks_per_gpu=0,
        )


def test_host_memory_budget_is_conservative_and_model_aware():
    crop = {"model_id": "CropGenomeFM"}
    public = {"model_id": "AgroNT_1B"}
    assert _host_memory_budget_for_group(crop, 2304) == 49152
    assert _host_memory_budget_for_group(public, 1536) == 8192
    assert _host_memory_budget_for_group(public, 10240) == 20480


def test_strict_gpu_plan_does_not_enable_memory_packing():
    assert _memory_policy_for_group({}, {
        "model_id": "model", "context_bp": 512, "mode": "pooled",
    }) is None


def test_terminal_group_failure_is_plan_and_log_bound(tmp_path):
    group = {"group_key": "g", "run_keys": ["r"]}
    log = tmp_path / "gpu_groups/g/worker.log"
    log.parent.mkdir(parents=True); log.write_text("trace")
    _write_terminal_group_failure(group, tmp_path, "plan", 3, "failed")
    assert _valid_terminal_group_failure(group, tmp_path, "plan") is not None
    assert _valid_terminal_group_failure(group, tmp_path, "other") is None
    log.write_text("tampered")
    assert _valid_terminal_group_failure(group, tmp_path, "plan") is None


def test_cpu_failure_becomes_terminal_after_three_hash_bound_attempts(tmp_path):
    row = {"run_key": "shared__T__model__ctx32"}
    assert record_cpu_row_failure(row, tmp_path, "plan", "one")["status"] == "failed"
    assert record_cpu_row_failure(row, tmp_path, "plan", "two")["status"] == "failed"
    assert record_cpu_row_failure(row, tmp_path, "plan", "three")["status"] == "terminal_failed"
    terminal = _valid_terminal_cpu_failure(row, tmp_path, "plan")
    assert terminal is not None and terminal["attempts"] == 3
    Path(terminal["artifacts"][0]["path"]).write_text("tampered")
    assert _valid_terminal_cpu_failure(row, tmp_path, "plan") is None


def test_cpu_kmer_row_closes_final_receipt(tmp_path):
    final_root = tmp_path / "final"
    dataset = final_root / "datasets" / "T"
    rows = []
    for index in range(60):
        split = "train" if index < 30 else "validation" if index < 45 else "test"
        rows.append({
            "sample_id": f"s{index}", "split": split,
            "sequence": ("ACGT" * 8) + ("A" * index) + "C",
            "label": index % 2, "species": "plant", "group_id": f"g{index}",
        })
    materialize_rows(rows, dataset, "T", "binary_classification", {"source": "test"})
    row = {
        "run_key": "shared/kmer_logistic/T/ctx32", "task_id": "T",
        "task_kind": "binary_classification", "split_policy": "fixed_train_validation_test",
        "cv_policy": "fixed_train_validation_test",
        "model_id": "kmer_logistic", "model_kind": "simple_baseline",
        "checkpoint_scope": "shared", "checkpoint_step": None, "context_bp": 32,
        "execution_kind": "evaluation",
    }
    receipt = execute_cpu_row(row, tmp_path, final_root, "plan-sha", seeds=[13, 29])
    assert receipt["status"] == "ok"
    assert receipt["plan_sha256"] == "plan-sha"
    assert receipt["test_metrics"]
    assert Path(receipt["result_receipt"]).is_file()
    result = json.loads(Path(receipt["result_receipt"]).read_text())
    assert result["seeds"] == [13, 29]
    assert Path(result["test_predictions"]).is_file()
