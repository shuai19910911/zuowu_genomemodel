import sys
import json
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.gpu_gate import (
    _host_memory_capacity_report, _process_tree_rss_mib, _telemetry_file_guard,
    acquire_memory_claim, active_memory_claims,
    leased_uuids, parse_gpu_state,
    preflight_gpu, query_gpu_state, release_memory_claim,
    run_with_dynamic_gpu, run_with_memory_packed_gpu,
    select_gpu, select_memory_packed_gpu,
)


def test_gpu_gate_blocks_compute_and_abnormal_memory_but_ignores_nvitop():
    gpu_csv = "\n".join([
        "0, GPU-a, NVIDIA GeForce RTX 2080 Ti, 11264, 1, 0",
        "1, GPU-b, NVIDIA GeForce RTX 2080 Ti, 11264, 1419, 56",
        "2, GPU-c, NVIDIA GeForce RTX 2080 Ti, 11264, 2, 0",
    ])
    app_csv = "\n".join([
        "GPU-b, 12, /env/python, 1416",
        "GPU-c, 99, /usr/bin/nvitop, 1",
    ])
    state = parse_gpu_state(gpu_csv, app_csv)
    selected = select_gpu(state, leased_uuids=set(), max_used_mib=512, max_utilization=10)
    assert selected["uuid"] == "GPU-a"
    selected = select_gpu(state, leased_uuids={"GPU-a"}, max_used_mib=512, max_utilization=10)
    assert selected["uuid"] == "GPU-c"


def test_gpu_gate_returns_none_when_every_card_is_blocked():
    state = parse_gpu_state("0, GPU-a, RTX, 11264, 900, 0", "")
    assert select_gpu(state, leased_uuids=set(), max_used_mib=512, max_utilization=10) is None


def test_gpu_gate_blocks_fuser_holders_but_not_nvitop():
    state = parse_gpu_state(
        "0, GPU-a, RTX, 11264, 1, 0\n1, GPU-b, RTX, 11264, 1, 0", "",
    )
    state[0]["fuser_holders"] = [{"pid": 10, "process_name": "/env/python", "used_memory_mib": 0}]
    state[1]["fuser_holders"] = [{"pid": 11, "process_name": "/env/nvitop", "used_memory_mib": 0}]
    assert select_gpu(state, set())["uuid"] == "GPU-b"
    state[1]["fuser_query_error"] = "permission denied"
    assert select_gpu(state, set()) is None


def test_gpu_gate_blocks_driver_not_ready_state():
    state = parse_gpu_state("\n".join([
        "3, GPU-good, RTX, 11264, 1, 0, P8, 0x0000000000000004",
        "4, GPU-bad, RTX, 11264, 1, 0, P0, System is not in ready state",
    ]), "")
    assert state[1]["system_ready"] is False
    assert select_gpu(state, {"GPU-good"}) is None


def test_dead_local_gpu_lease_is_removed_but_remote_host_lease_is_kept(tmp_path):
    dead = tmp_path / "GPU-dead"; dead.mkdir()
    (dead / "lease.json").write_text(json.dumps({"pid": 99999999, "host": socket.gethostname()}))
    remote = tmp_path / "GPU-remote"; remote.mkdir()
    (remote / "lease.json").write_text(json.dumps({"pid": 99999999, "host": "another-host"}))
    assert leased_uuids(tmp_path) == {"GPU-remote"}
    assert not dead.exists()


def test_dynamic_gpu_rechecks_authorization_after_lease(monkeypatch, tmp_path):
    monkeypatch.setattr("downstream_v4.gpu_gate.preflight_gpu", lambda *args, **kwargs: {
        "status": "ready", "selected": {"uuid": "GPU-test", "index": 0},
    })
    launched = []
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.subprocess.run",
        lambda *args, **kwargs: launched.append(True),
    )

    def revoked():
        raise RuntimeError("authorization revoked in test")

    with pytest.raises(RuntimeError, match="authorization revoked in test"):
        run_with_dynamic_gpu(
            ["forbidden-command"], "test", lease_root=tmp_path,
            before_launch=revoked,
        )
    assert launched == []
    assert not (tmp_path / "GPU-test").exists()


def test_exclusive_gpu_preflight_blocks_active_shared_memory_claim(
        monkeypatch, tmp_path):
    state = parse_gpu_state("0, GPU-shared, RTX, 11264, 1, 0", "")
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.query_gpu_state", lambda *args, **kwargs: state,
    )
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.leased_uuids", lambda *args, **kwargs: set(),
    )
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.active_memory_claims",
        lambda *args, **kwargs: {"GPU-shared": [{"memory_budget_mib": 3000}]},
    )
    report = preflight_gpu(
        lease_root=tmp_path / "leases", claim_root=tmp_path / "claims",
    )
    assert report["status"] == "blocked_no_free_gpu"
    assert report["leased_uuids"] == ["GPU-shared"]


def test_memory_packed_gpu_blocks_foreign_compute_even_when_budget_would_fit():
    state = parse_gpu_state(
        "0, GPU-a, RTX, 11264, 1, 0\n"
        "1, GPU-b, RTX, 11264, 1013, 80",
        "GPU-b, 123, /other-user/python, 1010",
    )
    selected = select_memory_packed_gpu(
        state, claims_by_uuid={"GPU-a": [{"memory_budget_mib": 7000}]},
        required_memory_mib=7000, reserved_headroom_mib=1024,
        max_tasks_per_gpu=1,
    )
    assert selected is None


def test_memory_packed_gpu_allows_foreign_compute_when_budget_and_headroom_fit():
    state = parse_gpu_state(
        "4, GPU-shared, RTX, 11264, 13, 80",
        "GPU-shared, 123, /other-user/python, 10",
    )
    selected = select_memory_packed_gpu(
        state, claims_by_uuid={}, required_memory_mib=7000,
        reserved_headroom_mib=1024, max_tasks_per_gpu=1,
        foreign_compute_allowed=True,
    )
    assert selected["uuid"] == "GPU-shared"
    assert selected["foreign_used_mib"] == 10
    assert selected["remaining_after_claim_mib"] == 4251


def test_memory_packed_gpu_uses_only_an_unclaimed_idle_card():
    state = parse_gpu_state(
        "0, GPU-a, RTX, 11264, 2001, 20\n"
        "1, GPU-b, RTX, 11264, 1, 0", "",
    )
    claims = {"GPU-a": [{"memory_budget_mib": 3000}], "GPU-b": []}
    selected = select_memory_packed_gpu(
        state, claims, required_memory_mib=3000,
        reserved_headroom_mib=1024, max_tasks_per_gpu=1,
    )
    assert selected["uuid"] == "GPU-b"

    claims["GPU-b"] = [{"memory_budget_mib": 3000}]
    selected = select_memory_packed_gpu(
        state, claims, required_memory_mib=3000,
        reserved_headroom_mib=1024, max_tasks_per_gpu=1,
    )
    assert selected is None


def test_memory_packed_gpu_allows_multiple_owned_claims_when_memory_fits():
    state = parse_gpu_state(
        "0, GPU-shared, RTX, 11264, 1001, 30",
        "GPU-shared, 111, /project/python, 1000",
    )
    claims = {"GPU-shared": [{
        "memory_budget_mib": 3000, "application_pids": [111],
    }]}
    state[0]["fuser_holders"] = [
        {"pid": 111, "process_name": "/project/python", "used_memory_mib": 0},
    ]
    selected = select_memory_packed_gpu(
        state, claims, required_memory_mib=3000,
        reserved_headroom_mib=1024, max_tasks_per_gpu=0,
        leased_gpu_uuids={"GPU-shared"},
    )
    assert selected["uuid"] == "GPU-shared"
    assert selected["active_claims"] == 1
    assert selected["remaining_after_claim_mib"] == 5263


def test_memory_packed_gpu_blocks_claim_that_would_consume_reserved_headroom():
    state = parse_gpu_state(
        "0, GPU-a, RTX, 11264, 4013, 0",
        "GPU-a, 123, /other-user/python, 4010",
    )
    assert select_memory_packed_gpu(
        state, claims_by_uuid={}, required_memory_mib=7000,
        reserved_headroom_mib=1024, max_tasks_per_gpu=1,
    ) is None


def test_memory_packed_gpu_blocks_regular_uuid_lease_and_fuser_holder():
    state = parse_gpu_state(
        "0, GPU-a, RTX, 11264, 1, 0\n1, GPU-b, RTX, 11264, 1, 0", "",
    )
    state[1]["fuser_holders"] = [
        {"pid": 55, "process_name": "/other/python", "used_memory_mib": 0},
    ]
    assert select_memory_packed_gpu(
        state, {}, required_memory_mib=3000,
        leased_gpu_uuids={"GPU-a"}, max_tasks_per_gpu=1,
    ) is None


def test_dynamic_gpu_rejects_wrong_required_hostname(tmp_path):
    required = "gpu05" if socket.gethostname() != "gpu05" else "not-gpu05"
    with pytest.raises(RuntimeError, match="GPU execution requires host"):
        run_with_dynamic_gpu(
            ["forbidden"], "test", lease_root=tmp_path,
            required_hostname=required,
        )


def test_memory_claims_allow_multiple_owned_tasks_on_one_uuid(tmp_path):
    first_path, _ = acquire_memory_claim(
        "GPU-a", "first", 3000, claim_root=tmp_path,
        host_memory_budget_mib=49152,
    )
    second_path, _ = acquire_memory_claim(
        "GPU-a", "second", 2500, claim_root=tmp_path,
    )
    claims = active_memory_claims(tmp_path)
    assert [row["memory_budget_mib"] for row in claims["GPU-a"]] == [2500, 3000]
    assert next(
        row for row in claims["GPU-a"] if row["purpose"] == "first"
    )["host_memory_budget_mib"] == 49152

    release_memory_claim(first_path)
    claims = active_memory_claims(tmp_path)
    assert len(claims["GPU-a"]) == 1
    release_memory_claim(second_path)
    assert active_memory_claims(tmp_path) == {}


def test_host_memory_gate_reserves_claim_budget_not_yet_in_child_rss():
    claims = {"GPU-a": [{
        "host_memory_budget_mib": 49152, "child_pid": 123,
    }]}
    ready = _host_memory_capacity_report(
        claims, required_host_memory_mib=49152,
        reserved_host_headroom_mib=8192,
        available_host_memory_mib=100000,
        process_rss_lookup=lambda pid: 10000,
    )
    blocked = _host_memory_capacity_report(
        claims, required_host_memory_mib=49152,
        reserved_host_headroom_mib=8192,
        available_host_memory_mib=90000,
        process_rss_lookup=lambda pid: 10000,
    )
    assert ready["status"] == "ready"
    assert ready["pending_claim_reservation_mib"] == 39152
    assert ready["remaining_after_claim_mib"] == 3504
    assert blocked["status"] == "blocked_host_memory_capacity"


def test_process_tree_rss_includes_all_descendants(tmp_path):
    rows = {
        100: (1, 1024),
        101: (100, 2048),
        102: (101, 3072),
        200: (1, 8192),
    }
    for pid, (ppid, rss_kib) in rows.items():
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "status").write_text(
            f"Name:\ttest\nPPid:\t{ppid}\nVmRSS:\t{rss_kib} kB\n"
        )
    assert _process_tree_rss_mib(100, proc_root=tmp_path) == pytest.approx(6.0)


def test_memory_claims_remove_dead_local_owner_but_keep_remote_owner(tmp_path):
    local = tmp_path / "GPU-a"; local.mkdir()
    (local / "dead.json").write_text(json.dumps({
        "claim_id": "dead", "gpu_uuid": "GPU-a", "pid": 99999999,
        "host": socket.gethostname(), "memory_budget_mib": 3000,
    }))
    remote = tmp_path / "GPU-b"; remote.mkdir()
    (remote / "remote.json").write_text(json.dumps({
        "claim_id": "remote", "gpu_uuid": "GPU-b", "pid": 99999999,
        "host": "another-host", "memory_budget_mib": 3000,
    }))
    claims = active_memory_claims(tmp_path)
    assert "GPU-a" not in claims
    assert len(claims["GPU-b"]) == 1
    assert not (local / "dead.json").exists()


def test_memory_packed_run_keeps_running_with_foreign_compute_when_allowed(
        monkeypatch, tmp_path):
    selected = {
        "uuid": "GPU-shared", "index": 4, "memory_total_mib": 11264,
        "memory_used_mib": 13, "utilization_percent": 80,
        "applications": [
            {"pid": 123, "process_name": "/other-user/python", "used_memory_mib": 10},
        ],
    }
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.preflight_memory_packed_gpu",
        lambda *args, **kwargs: {"status": "ready", "selected": selected},
    )
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.query_gpu_state",
        lambda *args, **kwargs: [selected],
    )
    result = run_with_memory_packed_gpu(
        [sys.executable, "-c", "import time; time.sleep(0.08)"],
        "test", memory_budget_mib=3000, claim_root=tmp_path,
        lease_root=tmp_path / "leases", foreign_compute_allowed=True,
        wait_timeout_seconds=1, poll_seconds=0.01, monitor_seconds=0.01,
    )
    assert result["returncode"] == 0
    assert result["safety_stop"] is None


def test_memory_packed_run_retries_preflight_telemetry_timeout(monkeypatch, tmp_path):
    selected = {
        "uuid": "GPU-test", "index": 6, "memory_total_mib": 11264,
        "memory_used_mib": 1, "applications": [],
    }
    calls = []

    def flaky_preflight(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise TimeoutError("telemetry lock busy")
        return {"status": "ready", "selected": selected}

    monkeypatch.setattr(
        "downstream_v4.gpu_gate.preflight_memory_packed_gpu", flaky_preflight,
    )
    result = run_with_memory_packed_gpu(
        [sys.executable, "-c", "pass"], "test", memory_budget_mib=3000,
        claim_root=tmp_path, lease_root=tmp_path / "leases",
        wait_timeout_seconds=1, poll_seconds=0.01, monitor_seconds=0.01,
    )
    assert result["returncode"] == 0
    assert len(calls) == 2
    assert result["preflight_telemetry_errors"] == 1


def test_memory_packed_run_waits_for_host_memory_capacity(monkeypatch, tmp_path):
    selected = {
        "uuid": "GPU-test", "index": 6, "memory_total_mib": 11264,
        "memory_used_mib": 1000, "applications": [],
    }
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.preflight_memory_packed_gpu",
        lambda *args, **kwargs: {"status": "ready", "selected": selected},
    )
    reports = iter([
        {"status": "blocked_host_memory_capacity"},
        {"status": "ready", "remaining_after_claim_mib": 1024},
    ])
    calls = []

    def host_capacity(*args, **kwargs):
        calls.append((args, kwargs))
        return next(reports)

    monkeypatch.setattr(
        "downstream_v4.gpu_gate._host_memory_capacity_report", host_capacity,
    )
    result = run_with_memory_packed_gpu(
        [sys.executable, "-c", "pass"], "test", memory_budget_mib=3000,
        host_memory_budget_mib=49152, reserved_host_headroom_mib=8192,
        claim_root=tmp_path / "claims", lease_root=tmp_path / "leases",
        wait_timeout_seconds=1, poll_seconds=0.01, monitor_seconds=0.01,
    )
    assert result["returncode"] == 0
    assert result["host_memory_budget_mib"] == 49152
    assert len(calls) == 2


def test_memory_packed_run_stops_process_tree_that_exceeds_host_budget(
        monkeypatch, tmp_path):
    selected = {
        "uuid": "GPU-test", "index": 6, "memory_total_mib": 11264,
        "memory_used_mib": 1000, "applications": [],
    }
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.preflight_memory_packed_gpu",
        lambda *args, **kwargs: {"status": "ready", "selected": selected},
    )
    monkeypatch.setattr(
        "downstream_v4.gpu_gate._host_memory_capacity_report",
        lambda *args, **kwargs: {"status": "ready"},
    )
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.query_gpu_state",
        lambda *args, **kwargs: [selected],
    )
    monkeypatch.setattr(
        "downstream_v4.gpu_gate._process_tree_rss_mib", lambda pid: 4000.0,
    )
    result = run_with_memory_packed_gpu(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        "host-over-budget", memory_budget_mib=3000,
        host_memory_budget_mib=3000, reserved_host_headroom_mib=512,
        claim_root=tmp_path / "claims", lease_root=tmp_path / "leases",
        wait_timeout_seconds=1, poll_seconds=0.01, monitor_seconds=0.01,
    )
    assert result["returncode"] != 0
    assert result["safety_stop"]["reason"] == "host_memory_budget_exceeded"
    assert result["peak_host_process_tree_memory_mib"] == pytest.approx(4000.0)


def test_memory_packed_run_uses_shared_claim_without_replacing_legacy_lease(
        monkeypatch, tmp_path):
    selected = {
        "uuid": "GPU-test", "index": 6, "memory_total_mib": 11264,
        "memory_used_mib": 1000, "applications": [],
    }
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.preflight_memory_packed_gpu",
        lambda *args, **kwargs: {"status": "ready", "selected": selected},
    )
    lease_root = tmp_path / "leases"
    legacy = lease_root / "GPU-test"
    legacy.mkdir(parents=True)
    receipt = legacy / "lease.json"
    receipt.write_text(json.dumps({
        "gpu_uuid": "GPU-test", "purpose": "existing", "pid": 1,
        "host": "remote-host",
    }))
    result = run_with_memory_packed_gpu(
        [sys.executable, "-c", "pass"], "test", memory_budget_mib=3000,
        claim_root=tmp_path / "claims", lease_root=lease_root,
        wait_timeout_seconds=1, poll_seconds=0.01, monitor_seconds=0.01,
    )
    assert result["returncode"] == 0
    assert result["lease"]["mode"] == "shared_memory_claim"
    assert receipt.is_file()
    assert active_memory_claims(tmp_path / "claims") == {}


def test_memory_packed_run_binds_selected_uuid_and_releases_claim(monkeypatch, tmp_path):
    selected = {
        "uuid": "GPU-test", "index": 3, "memory_total_mib": 11264,
        "memory_used_mib": 1, "applications": [],
    }
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.preflight_memory_packed_gpu",
        lambda *args, **kwargs: {"status": "ready", "selected": selected},
    )
    result = run_with_memory_packed_gpu(
        [sys.executable, "-c", "import os; assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-test'"],
        "test", memory_budget_mib=3000, claim_root=tmp_path,
        lease_root=tmp_path / "leases",
        wait_timeout_seconds=1, poll_seconds=0.01, monitor_seconds=0.01,
    )
    assert result["returncode"] == 0
    assert result["gpu"]["uuid"] == "GPU-test"
    assert result["memory_budget_mib"] == 3000
    assert active_memory_claims(tmp_path) == {}
    assert leased_uuids(tmp_path / "leases") == set()


def test_memory_packed_run_releases_claim_when_authorization_is_revoked(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "downstream_v4.gpu_gate.preflight_memory_packed_gpu",
        lambda *args, **kwargs: {
            "status": "ready", "selected": {
                "uuid": "GPU-test", "index": 0, "memory_total_mib": 11264,
                "memory_used_mib": 1, "applications": [],
            },
        },
    )
    with pytest.raises(RuntimeError, match="revoked"):
        run_with_memory_packed_gpu(
            ["forbidden"], "test", memory_budget_mib=3000,
            claim_root=tmp_path, lease_root=tmp_path / "leases",
            wait_timeout_seconds=1,
            before_launch=lambda: (_ for _ in ()).throw(RuntimeError("revoked")),
        )
    assert active_memory_claims(tmp_path) == {}
    assert leased_uuids(tmp_path / "leases") == set()


def test_gpu_telemetry_queries_are_serialized_across_workers(monkeypatch, tmp_path):
    active = 0
    maximum = 0
    guard = threading.Lock()

    def fake_run(argv, **kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        stdout = (
            "0, GPU-a, RTX, 11264, 1, 0, P8, 0x0\n"
            if "--query-gpu" in argv[1] else ""
        )
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("downstream_v4.gpu_gate.subprocess.run", fake_run)
    monkeypatch.setattr("downstream_v4.gpu_gate.shutil.which", lambda name: None)
    lock = tmp_path / "telemetry.lock"
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(
            lambda _: query_gpu_state(telemetry_lock_path=lock), range(2),
        ))
    assert len(rows) == 2
    assert maximum == 1


def test_basic_gpu_telemetry_skips_hanging_compute_application_query(monkeypatch, tmp_path):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(
            argv, 0,
            stdout="0, GPU-a, RTX, 11264, 3, 0, P8, 0x0\n",
            stderr="",
        )

    monkeypatch.setattr("downstream_v4.gpu_gate.subprocess.run", fake_run)
    monkeypatch.setattr("downstream_v4.gpu_gate.shutil.which", lambda name: None)
    state = query_gpu_state(
        telemetry_lock_path=tmp_path / "telemetry.lock",
        include_applications=False,
        cache_max_age_seconds=0,
    )
    assert len(calls) == 1
    assert "--query-gpu" in calls[0][1]
    assert state[0]["applications"] == []
    assert state[0]["application_telemetry_available"] is False


def test_gpu_telemetry_file_guard_is_reentrant_for_same_process(tmp_path):
    lock = tmp_path / "telemetry.lock"
    with _telemetry_file_guard(lock, timeout_seconds=1):
        with _telemetry_file_guard(lock, timeout_seconds=1):
            assert True


def test_gpu_telemetry_reuses_fresh_cross_process_cache(monkeypatch, tmp_path):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv[1])
        stdout = (
            "0, GPU-a, RTX, 11264, 1, 0, P8, 0x0\n"
            if "--query-gpu" in argv[1] else ""
        )
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("downstream_v4.gpu_gate.subprocess.run", fake_run)
    monkeypatch.setattr("downstream_v4.gpu_gate.shutil.which", lambda name: None)
    lock = tmp_path / "telemetry.lock"
    cache = tmp_path / "telemetry.cache.json"
    first = query_gpu_state(
        telemetry_lock_path=lock, telemetry_cache_path=cache,
        cache_max_age_seconds=1,
    )
    second = query_gpu_state(
        telemetry_lock_path=lock, telemetry_cache_path=cache,
        cache_max_age_seconds=1,
    )
    assert first == second
    assert len(calls) == 2
