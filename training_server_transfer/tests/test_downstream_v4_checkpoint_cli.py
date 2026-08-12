import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/launch_cropgenome_downstream_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("article_checkpoint_cli", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_matching_heartbeat_requires_plan_host_and_pid(tmp_path):
    heartbeat = tmp_path / "controller/GPU_DAEMON_HEARTBEAT.json"
    heartbeat.parent.mkdir(parents=True)
    heartbeat.write_text(json.dumps({
        "status": "running",
        "plan_sha256": "a" * 64,
        "hostname": "gpu05",
        "pid": 1234,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }))
    payload = MODULE._wait_for_matching_heartbeat(
        tmp_path, "a" * 64, timeout_seconds=0.01,
    )
    assert payload["pid"] == 1234


def test_matching_heartbeat_fails_closed_on_wrong_plan(tmp_path, monkeypatch):
    heartbeat = tmp_path / "controller/GPU_DAEMON_HEARTBEAT.json"
    heartbeat.parent.mkdir(parents=True)
    heartbeat.write_text(json.dumps({
        "status": "running",
        "plan_sha256": "b" * 64,
        "hostname": "gpu05",
        "pid": 1234,
    }))
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="matching live heartbeat"):
        MODULE._wait_for_matching_heartbeat(
            tmp_path, "a" * 64, timeout_seconds=0.001,
        )


def test_matching_heartbeat_rejects_stale_same_plan(tmp_path, monkeypatch):
    heartbeat = tmp_path / "controller/GPU_DAEMON_HEARTBEAT.json"
    heartbeat.parent.mkdir(parents=True)
    heartbeat.write_text(json.dumps({
        "status": "running",
        "plan_sha256": "a" * 64,
        "hostname": "gpu05",
        "pid": 1234,
        "updated_at": "2026-08-12T00:00:00+00:00",
    }))
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="matching live heartbeat"):
        MODULE._wait_for_matching_heartbeat(
            tmp_path, "a" * 64, timeout_seconds=0.001,
            not_before=datetime(2026, 8, 12, 1, 0, tzinfo=timezone.utc),
        )


def test_gpu_host_resolution_prefers_env_then_local_file(tmp_path, monkeypatch):
    local = tmp_path / "controller/GPU_HOST.local.json"
    local.parent.mkdir(parents=True)
    local.write_text(json.dumps({"status": "active", "host": "local-host"}))
    monkeypatch.delenv("CROPGENOME_GPU_HOST", raising=False)
    assert MODULE._resolve_gpu_host(tmp_path) == "local-host"
    monkeypatch.setenv("CROPGENOME_GPU_HOST", "env-host")
    assert MODULE._resolve_gpu_host(tmp_path) == "env-host"
    monkeypatch.setenv("CROPGENOME_GPU_HOST", "-oProxyCommand=bad")
    with pytest.raises(RuntimeError, match="invalid GPU SSH host"):
        MODULE._resolve_gpu_host(tmp_path)


def test_restore_archived_or_remove_deletes_new_files_without_backups(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    plan = tmp_path / "FINAL_PROTOCOL.json"
    auth = tmp_path / "FORMAL_EXECUTION_AUTHORIZATION.json"
    plan.write_text("new-plan")
    auth.write_text("new-auth")
    MODULE._restore_archived_or_remove(archive, (plan, auth))
    assert not plan.exists()
    assert not auth.exists()
