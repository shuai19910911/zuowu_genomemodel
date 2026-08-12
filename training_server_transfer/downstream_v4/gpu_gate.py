"""Dynamic NVIDIA GPU preflight and atomic UUID leases."""

import csv
import fcntl
import io
import json
import os
import signal
import shutil
import socket
import subprocess
import re
import time
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


LEASE_ROOT = Path("/tmp/cropgenome_gpu_leases")
MEMORY_CLAIM_ROOT = Path("/tmp/cropgenome_gpu_memory_claims")
TELEMETRY_LOCK_PATH = Path("/tmp/cropgenome_gpu_telemetry.lock")
PREFLIGHT_LOCK_PATH = Path("/tmp/cropgenome_gpu_preflight.lock")
TELEMETRY_ERRORS = (OSError, ValueError, subprocess.SubprocessError, TimeoutError)


def _csv_rows(text):
    return [[value.strip() for value in row] for row in csv.reader(io.StringIO(text)) if row]


def parse_gpu_state(gpu_csv, application_csv):
    gpus = []
    by_uuid = {}
    for fields in _csv_rows(gpu_csv):
        if len(fields) < 6:
            raise ValueError(f"malformed nvidia-smi GPU row: {fields}")
        row = {
            "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
            "memory_total_mib": int(fields[3].split()[0]),
            "memory_used_mib": int(fields[4].split()[0]),
            "utilization_percent": int(fields[5].split()[0]), "applications": [],
            "pstate": fields[6] if len(fields) >= 7 else None,
            "health_detail": fields[7] if len(fields) >= 8 else None,
            "system_ready": (
                "not in ready state" not in fields[7].lower()
                if len(fields) >= 8 else True
            ),
        }
        gpus.append(row); by_uuid[row["uuid"]] = row
    for fields in _csv_rows(application_csv):
        if len(fields) < 4 or fields[0] not in by_uuid:
            continue
        application = {
            "pid": int(fields[1]), "process_name": fields[2],
            "used_memory_mib": int(fields[3].split()[0]),
        }
        by_uuid[fields[0]]["applications"].append(application)
    return sorted(gpus, key=lambda row: row["index"])


def _is_benign_monitor(application):
    return "nvitop" in Path(application["process_name"]).name.lower()


def select_gpu(gpu_state, leased_uuids, max_used_mib=512, max_utilization=10,
               allowed_indices=None):
    allowed = None if allowed_indices is None else {int(value) for value in allowed_indices}
    candidates = []
    for gpu in gpu_state:
        if allowed is not None and int(gpu["index"]) not in allowed:
            continue
        if not gpu.get("system_ready", True):
            continue
        blockers = [app for app in gpu["applications"] if not _is_benign_monitor(app)]
        fuser_blockers = [app for app in gpu.get("fuser_holders", []) if not _is_benign_monitor(app)]
        if gpu["uuid"] in leased_uuids or blockers or fuser_blockers or gpu.get("fuser_query_error"):
            continue
        if gpu["memory_used_mib"] > max_used_mib or gpu["utilization_percent"] > max_utilization:
            continue
        candidates.append(gpu)
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["memory_used_mib"], row["utilization_percent"], row["index"]))


def select_memory_packed_gpu(gpu_state, claims_by_uuid, required_memory_mib,
                             reserved_headroom_mib=1024, max_tasks_per_gpu=1,
                             leased_gpu_uuids=None, allowed_indices=None,
                             max_unaccounted_used_mib=512,
                             max_unclaimed_utilization=10,
                             foreign_compute_allowed=False):
    """Select a GPU from conservative owned claims and observed foreign memory.

    Active owned claims reserve their full conservative budgets. Unknown compute,
    unknown device holders, and anomalous unaccounted memory are always fail-closed.
    Explicit sharing applies only to process-level compute with measurable memory.
    """
    required = int(required_memory_mib)
    headroom = int(reserved_headroom_mib)
    limit = int(max_tasks_per_gpu)
    leased = set(leased_gpu_uuids or ())
    allowed = None if allowed_indices is None else {int(value) for value in allowed_indices}
    if required <= 0 or headroom < 0 or not 1 <= limit <= 3:
        raise ValueError("memory budgets must be positive and task limit between 1 and 3")
    candidates = []
    for gpu in gpu_state:
        if allowed is not None and int(gpu["index"]) not in allowed:
            continue
        if not gpu.get("system_ready", True):
            continue
        if gpu.get("application_telemetry_available") is False:
            continue
        claims = list((claims_by_uuid or {}).get(gpu["uuid"], []))
        if gpu["uuid"] in leased and not claims:
            continue
        if limit and len(claims) >= limit:
            continue
        owned_pids = {
            int(pid) for claim in claims for pid in claim.get("application_pids", [])
        }
        applications = [
            app for app in gpu.get("applications", []) if not _is_benign_monitor(app)
        ]
        application_pids = {int(app.get("pid", -1)) for app in applications}
        fuser_blockers = [
            app for app in gpu.get("fuser_holders", [])
            if not _is_benign_monitor(app)
            and int(app.get("pid", -1)) not in owned_pids | application_pids
        ]
        if fuser_blockers or gpu.get("fuser_query_error"):
            continue
        foreign_applications = [
            app for app in applications if int(app.get("pid", -1)) not in owned_pids
        ]
        if foreign_applications and not foreign_compute_allowed:
            continue
        claim_over_budget = any(
            sum(
                int(app.get("used_memory_mib", 0))
                for app in applications
                if int(app.get("pid", -1))
                in {int(pid) for pid in claim.get("application_pids", [])}
            ) > int(claim["memory_budget_mib"])
            for claim in claims
        )
        if claim_over_budget:
            continue
        application_used = sum(int(app.get("used_memory_mib", 0)) for app in applications)
        foreign_used = sum(
            int(app.get("used_memory_mib", 0)) for app in foreign_applications
        )
        unaccounted_used = max(0, int(gpu["memory_used_mib"]) - application_used)
        if unaccounted_used > int(max_unaccounted_used_mib):
            continue
        if (
            not foreign_compute_allowed and not claims
            and int(gpu["utilization_percent"]) > int(max_unclaimed_utilization)
        ):
            continue
        committed = sum(int(claim["memory_budget_mib"]) for claim in claims)
        remaining = (
            int(gpu["memory_total_mib"]) - foreign_used - unaccounted_used
            - committed - required
        )
        if remaining < headroom:
            continue
        candidate = dict(gpu)
        candidate.update({
            "active_claims": len(claims),
            "foreign_used_mib": foreign_used,
            "foreign_compute_allowed": bool(foreign_compute_allowed),
            "unaccounted_used_mib": unaccounted_used,
            "committed_memory_mib": committed,
            "required_memory_mib": required,
            "remaining_after_claim_mib": remaining,
        })
        candidates.append(candidate)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            row["active_claims"], -row["remaining_after_claim_mib"], row["index"],
        ),
    )


def active_memory_claims(claim_root=MEMORY_CLAIM_ROOT):
    root = Path(claim_root)
    if not root.is_dir():
        return {}
    current_host = socket.gethostname()
    claims = {}
    for gpu_root in sorted(path for path in root.iterdir() if path.is_dir()):
        active = []
        for path in sorted(gpu_root.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                pid = int(record["pid"])
                same_host = record.get("host") == current_host
            except (OSError, ValueError, KeyError, TypeError):
                continue
            alive = True
            if same_host:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    alive = True
            if same_host and not alive:
                path.unlink(missing_ok=True)
                continue
            record["claim_path"] = str(path)
            active.append(record)
        if active:
            claims[gpu_root.name] = sorted(
                active,
                key=lambda row: (int(row["memory_budget_mib"]), str(row["claim_id"])),
            )
        else:
            try:
                gpu_root.rmdir()
            except OSError:
                pass
    return claims


def acquire_memory_claim(gpu_uuid, purpose, memory_budget_mib,
                         claim_root=MEMORY_CLAIM_ROOT,
                         host_memory_budget_mib=0):
    budget = int(memory_budget_mib)
    host_budget = int(host_memory_budget_mib)
    if budget <= 0 or host_budget < 0:
        raise ValueError("GPU memory budget must be positive and host budget non-negative")
    gpu_root = Path(claim_root) / str(gpu_uuid)
    gpu_root.mkdir(parents=True, exist_ok=True)
    claim_id = f"{os.getpid()}-{uuid.uuid4().hex}"
    path = gpu_root / f"{claim_id}.json"
    record = {
        "claim_id": claim_id, "gpu_uuid": str(gpu_uuid),
        "purpose": str(purpose), "pid": os.getpid(),
        "host": socket.gethostname(), "memory_budget_mib": budget,
        "host_memory_budget_mib": host_budget,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    return path, record


def release_memory_claim(path):
    path = Path(path)
    parent = path.parent
    path.unlink(missing_ok=True)
    try:
        parent.rmdir()
    except OSError:
        pass


def _process_rss_mib(pid):
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text(
                encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _process_tree_rss_mib(pid, proc_root="/proc"):
    """Return resident memory for a process and every live descendant."""
    root_pid = int(pid)
    processes = {}
    for status_path in Path(proc_root).glob("[0-9]*/status"):
        try:
            current_pid = int(status_path.parent.name)
            fields = {}
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith(("PPid:", "VmRSS:")):
                    key, value = line.split(":", 1)
                    fields[key] = int(value.split()[0])
            processes[current_pid] = (
                int(fields.get("PPid", 0)), int(fields.get("VmRSS", 0)),
            )
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for current_pid, (parent_pid, _) in processes.items():
            if current_pid not in descendants and parent_pid in descendants:
                descendants.add(current_pid)
                changed = True
    return sum(processes.get(current_pid, (0, 0))[1]
               for current_pid in descendants) / 1024.0


def _available_host_memory_mib(meminfo_path="/proc/meminfo"):
    for line in Path(meminfo_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _host_memory_capacity_report(claims_by_uuid, required_host_memory_mib=0,
                                 reserved_host_headroom_mib=8192,
                                 available_host_memory_mib=None,
                                 process_rss_lookup=_process_tree_rss_mib):
    required = int(required_host_memory_mib)
    headroom = int(reserved_host_headroom_mib)
    if required < 0 or headroom < 0:
        raise ValueError("host memory budget and headroom must be non-negative")
    available = (
        _available_host_memory_mib()
        if available_host_memory_mib is None else int(available_host_memory_mib)
    )
    pending = 0
    for claims in (claims_by_uuid or {}).values():
        for claim in claims:
            budget = max(0, int(claim.get("host_memory_budget_mib", 0)))
            child_pid = claim.get("child_pid")
            resident = int(process_rss_lookup(child_pid)) if child_pid else 0
            pending += max(0, budget - resident)
    remaining = available - pending - headroom - required
    return {
        "status": "ready" if remaining >= 0 else "blocked_host_memory_capacity",
        "available_host_memory_mib": available,
        "pending_claim_reservation_mib": pending,
        "required_host_memory_mib": required,
        "reserved_host_headroom_mib": headroom,
        "remaining_after_claim_mib": remaining,
    }


def _pid_descends_from(pid, ancestor_pid):
    pid = int(pid); ancestor_pid = int(ancestor_pid)
    for _ in range(64):
        if pid == ancestor_pid:
            return True
        if pid <= 1:
            return False
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
            pid = int(fields[3])
        except (OSError, ValueError, IndexError):
            return False
    return False


def _attach_claim_application_pids(claims, gpu_state):
    applications = [app for gpu in gpu_state for app in gpu.get("applications", [])]
    for records in claims.values():
        for record in records:
            owners = [int(record["pid"])]
            if record.get("child_pid") is not None:
                owners.append(int(record["child_pid"]))
            record["application_pids"] = sorted({
                int(app["pid"]) for app in applications
                if any(_pid_descends_from(app["pid"], owner) for owner in owners)
            })
    return claims


def preflight_memory_packed_gpu(memory_budget_mib, reserved_headroom_mib=1024,
                                max_tasks_per_gpu=1,
                                claim_root=MEMORY_CLAIM_ROOT,
                                lease_root=LEASE_ROOT, allowed_indices=None,
                                foreign_compute_allowed=False):
    state = query_gpu_state(cache_max_age_seconds=5, include_applications=True)
    claims = _attach_claim_application_pids(active_memory_claims(claim_root), state)
    leased = leased_uuids(lease_root)
    selected = select_memory_packed_gpu(
        state, claims, required_memory_mib=memory_budget_mib,
        reserved_headroom_mib=reserved_headroom_mib,
        max_tasks_per_gpu=max_tasks_per_gpu, leased_gpu_uuids=leased,
        allowed_indices=allowed_indices, max_unaccounted_used_mib=8,
        foreign_compute_allowed=foreign_compute_allowed,
    )
    return {
        "status": "ready" if selected else "blocked_no_memory_capacity",
        "selected": selected, "gpus": state, "claims_by_uuid": claims,
        "leased_uuids": sorted(leased),
        "policy": {
            "memory_budget_mib": int(memory_budget_mib),
            "reserved_headroom_mib": int(reserved_headroom_mib),
            "max_tasks_per_gpu": int(max_tasks_per_gpu),
            "foreign_compute_allowed": bool(foreign_compute_allowed),
            "unknown_compute_blocks": True,
            "nvitop_is_benign": True,
        },
    }


def _update_memory_claim(path, **updates):
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(updates)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    return record


def _stop_owned_process_group(process, grace_seconds=10):
    if process.poll() is not None:
        return process.returncode
    os.killpg(process.pid, signal.SIGTERM)
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.wait()


def run_with_memory_packed_gpu(argv, purpose, memory_budget_mib,
                               reserved_headroom_mib=1024, max_tasks_per_gpu=1,
                               host_memory_budget_mib=0,
                               reserved_host_headroom_mib=8192,
                               claim_root=MEMORY_CLAIM_ROOT,
                               lease_root=LEASE_ROOT, cwd=None,
                               extra_env=None, before_launch=None,
                               wait_timeout_seconds=3600, poll_seconds=15,
                               monitor_seconds=2, minimum_runtime_headroom_mib=256,
                               required_hostname=None, allowed_gpu_indices=None,
                               foreign_compute_allowed=False):
    if required_hostname is not None and socket.gethostname() != str(required_hostname):
        raise RuntimeError(
            f"GPU execution requires host {required_hostname}, got {socket.gethostname()}"
        )
    claim_root = Path(claim_root); claim_root.mkdir(parents=True, exist_ok=True)
    lease_root = Path(lease_root); lease_root.mkdir(parents=True, exist_ok=True)
    lock_path = PREFLIGHT_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + float(wait_timeout_seconds)
    claim_path = None; lease_path = None; selected = None; claim = None; lease = None
    preflight_telemetry_errors = 0; host_memory_report = None
    while selected is None:
        with lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                claims_snapshot = active_memory_claims(claim_root)
                host_memory_report = _host_memory_capacity_report(
                    claims_snapshot,
                    required_host_memory_mib=host_memory_budget_mib,
                    reserved_host_headroom_mib=reserved_host_headroom_mib,
                )
                if host_memory_report["status"] != "ready":
                    report = {**host_memory_report, "selected": None}
                else:
                    try:
                        report = preflight_memory_packed_gpu(
                            memory_budget_mib, reserved_headroom_mib,
                            max_tasks_per_gpu, claim_root, lease_root,
                            allowed_gpu_indices,
                            foreign_compute_allowed=foreign_compute_allowed,
                        )
                    except TELEMETRY_ERRORS:
                        preflight_telemetry_errors += 1
                        report = {"status": "telemetry_retry", "selected": None}
                selected = report.get("selected")
                if selected is not None:
                    claim_path, claim = acquire_memory_claim(
                        selected["uuid"], purpose, memory_budget_mib,
                        claim_root=claim_root,
                        host_memory_budget_mib=host_memory_budget_mib,
                    )
                    lease = {
                        "mode": "shared_memory_claim",
                        "gpu_uuid": selected["uuid"],
                        "claim_id": claim["claim_id"],
                        "purpose": str(purpose),
                    }
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        if selected is not None:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out waiting for memory-safe GPU capacity")
        time.sleep(float(poll_seconds))

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
    environment.update(extra_env or {})
    process = None; peak_memory = 0; telemetry_errors = 0; safety_stop = None
    peak_host_process_tree_memory_mib = 0.0
    try:
        if before_launch is not None:
            before_launch()
        process = subprocess.Popen(
            list(argv), cwd=cwd, env=environment, start_new_session=True,
        )
        claim = _update_memory_claim(claim_path, child_pid=process.pid)
        low_headroom_polls = 0
        gpu_over_budget_polls = 0
        host_over_budget_polls = 0
        while process.poll() is None:
            time.sleep(float(monitor_seconds))
            if process.poll() is not None:
                break
            try:
                host_process_tree_memory_mib = _process_tree_rss_mib(process.pid)
                peak_host_process_tree_memory_mib = max(
                    peak_host_process_tree_memory_mib,
                    host_process_tree_memory_mib,
                )
                host_over_budget_polls = (
                    host_over_budget_polls + 1
                    if int(host_memory_budget_mib) > 0
                    and host_process_tree_memory_mib > int(host_memory_budget_mib)
                    else 0
                )
                if host_over_budget_polls >= 2:
                    safety_stop = {
                        "reason": "host_memory_budget_exceeded",
                        "observed_process_tree_mib": host_process_tree_memory_mib,
                        "budget_mib": int(host_memory_budget_mib),
                    }
                    _stop_owned_process_group(process)
                    break
            except (OSError, ValueError):
                telemetry_errors += 1
            try:
                state = query_gpu_state(
                    cache_max_age_seconds=5, include_applications=True,
                )
                gpu = next(row for row in state if row["uuid"] == selected["uuid"])
                foreign = [
                    app for app in gpu.get("applications", [])
                    if not _is_benign_monitor(app)
                    and not _pid_descends_from(app["pid"], process.pid)
                    and not _pid_descends_from(app["pid"], os.getpid())
                ]
                if foreign and not foreign_compute_allowed:
                    safety_stop = "foreign_compute_detected_after_launch"
                    _stop_owned_process_group(process)
                    break
                owned = sum(
                    int(app.get("used_memory_mib", 0)) for app in gpu.get("applications", [])
                    if _pid_descends_from(app["pid"], process.pid)
                    or _pid_descends_from(app["pid"], os.getpid())
                )
                if not gpu.get("application_telemetry_available", True):
                    owned = max(
                        0,
                        int(gpu["memory_used_mib"])
                        - int(selected.get("memory_used_mib", 0)),
                    )
                peak_memory = max(peak_memory, owned)
                gpu_over_budget_polls = (
                    gpu_over_budget_polls + 1
                    if owned > int(memory_budget_mib)
                    else 0
                )
                if gpu_over_budget_polls >= 2:
                    safety_stop = {
                        "reason": "gpu_memory_budget_exceeded",
                        "observed_process_mib": int(owned),
                        "budget_mib": int(memory_budget_mib),
                    }
                    _stop_owned_process_group(process)
                    break
                free = int(gpu["memory_total_mib"]) - int(gpu["memory_used_mib"])
                low_headroom_polls = low_headroom_polls + 1 if free < int(minimum_runtime_headroom_mib) else 0
                if low_headroom_polls >= 2:
                    safety_stop = f"runtime_headroom_below_{int(minimum_runtime_headroom_mib)}_mib"
                    _stop_owned_process_group(process)
                    break
            except (OSError, ValueError, subprocess.SubprocessError, StopIteration):
                telemetry_errors += 1
        returncode = process.wait()
        return {
            "returncode": int(returncode), "gpu": selected, "claim": claim,
            "lease": lease,
            "memory_budget_mib": int(memory_budget_mib),
            "reserved_headroom_mib": int(reserved_headroom_mib),
            "host_memory_budget_mib": int(host_memory_budget_mib),
            "reserved_host_headroom_mib": int(reserved_host_headroom_mib),
            "host_memory_preflight": host_memory_report,
            "foreign_compute_allowed": bool(foreign_compute_allowed),
            "peak_process_memory_mib": int(peak_memory),
            "peak_host_process_tree_memory_mib": peak_host_process_tree_memory_mib,
            "preflight_telemetry_errors": int(preflight_telemetry_errors),
            "telemetry_errors": int(telemetry_errors), "safety_stop": safety_stop,
        }
    finally:
        if process is not None and process.poll() is None:
            _stop_owned_process_group(process)
        if claim_path is not None:
            release_memory_claim(claim_path)
        if lease_path is not None:
            release_gpu_lease(lease_path)


_TELEMETRY_PROCESS_GUARD = threading.RLock()
_TELEMETRY_PROCESS_DEPTH = 0


@contextmanager
def _telemetry_file_guard(lock_path, timeout_seconds=120):
    """Serialize NVIDIA telemetry while remaining reentrant in one process."""
    global _TELEMETRY_PROCESS_DEPTH
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _TELEMETRY_PROCESS_GUARD:
        if _TELEMETRY_PROCESS_DEPTH:
            _TELEMETRY_PROCESS_DEPTH += 1
            try:
                yield
            finally:
                _TELEMETRY_PROCESS_DEPTH -= 1
            return
        deadline = time.monotonic() + float(timeout_seconds)
        with lock_path.open("a+") as lock_handle:
            while True:
                try:
                    fcntl.flock(
                        lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out waiting for GPU telemetry lock")
                    time.sleep(0.05)
            _TELEMETRY_PROCESS_DEPTH = 1
            try:
                yield
            finally:
                _TELEMETRY_PROCESS_DEPTH = 0
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _read_telemetry_cache(path, max_age_seconds):
    path = Path(path)
    try:
        if time.time() - path.stat().st_mtime > float(max_age_seconds):
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = payload.get("state")
        return state if isinstance(state, list) else None
    except (OSError, ValueError, TypeError):
        return None


def _write_telemetry_cache(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "pid": os.getpid(), "state": state,
    }, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def query_gpu_state(timeout=90, telemetry_lock_path=TELEMETRY_LOCK_PATH,
                    telemetry_cache_path=None, cache_max_age_seconds=1,
                    include_applications=True):
    lock_path = Path(telemetry_lock_path)
    cache_path = (
        Path(telemetry_cache_path) if telemetry_cache_path is not None
        else Path(
            str(lock_path)
            + (".cache.json" if include_applications else ".basic.cache.json")
        )
    )
    cached = _read_telemetry_cache(cache_path, cache_max_age_seconds)
    if cached is not None:
        return cached
    with _telemetry_file_guard(lock_path, timeout_seconds=max(120, int(timeout))):
        cached = _read_telemetry_cache(cache_path, cache_max_age_seconds)
        if cached is not None:
            return cached
        gpu = subprocess.run([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pstate,clocks_throttle_reasons.active",
            "--format=csv,noheader,nounits",
        ], text=True, capture_output=True, timeout=timeout, check=True)
        application_csv = ""
        application_telemetry_available = False
        if include_applications:
            try:
                apps = subprocess.run([
                    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ], text=True, capture_output=True, timeout=min(int(timeout), 5), check=True)
                application_csv = apps.stdout
                application_telemetry_available = True
            except TELEMETRY_ERRORS:
                application_csv = ""
        state = parse_gpu_state(gpu.stdout, application_csv)
        executable = shutil.which("fuser")
        for row in state:
            row["application_telemetry_available"] = application_telemetry_available
            row["fuser_holders"] = []
            row["fuser_query_error"] = None
            if executable:
                holders, error = query_fuser_holders(row["index"], executable, timeout)
                row["fuser_holders"] = holders
                row["fuser_query_error"] = error
        _write_telemetry_cache(cache_path, state)
        return state


def query_fuser_holders(gpu_index, executable=None, timeout=30):
    executable = executable or shutil.which("fuser")
    if not executable:
        return [], None
    completed = subprocess.run(
        [executable, f"/dev/nvidia{int(gpu_index)}"], text=True,
        capture_output=True, timeout=timeout, check=False,
    )
    if completed.returncode not in {0, 1}:
        return [], f"fuser exited {completed.returncode}: {(completed.stderr or completed.stdout).strip()}"
    pids = sorted({int(value) for value in re.findall(r"\d+", completed.stdout)})
    holders = []
    for pid in pids:
        command = "unknown"
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            if raw:
                command = raw
            else:
                command = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
        except OSError:
            pass
        holders.append({"pid": pid, "process_name": command, "used_memory_mib": 0})
    return holders, None


def leased_uuids(lease_root=LEASE_ROOT):
    root = Path(lease_root)
    if not root.is_dir():
        return set()
    leased = set()
    local_host = socket.gethostname()
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            receipt = json.loads((path / "lease.json").read_text(encoding="utf-8"))
            pid = int(receipt["pid"])
            same_host = receipt.get("host") == local_host
        except (OSError, ValueError, KeyError, TypeError):
            leased.add(path.name)
            continue
        alive = True
        if same_host:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
        if same_host and not alive:
            shutil.rmtree(path)
            continue
        leased.add(path.name)
    return leased


def acquire_gpu_lease(gpu_uuid, purpose, lease_root=LEASE_ROOT):
    root = Path(lease_root); root.mkdir(parents=True, exist_ok=True)
    path = root / gpu_uuid
    try:
        path.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"GPU UUID already leased: {gpu_uuid}") from error
    receipt = {
        "gpu_uuid": gpu_uuid, "purpose": purpose, "pid": os.getpid(),
        "host": socket.gethostname(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (path / "lease.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, receipt


def release_gpu_lease(path):
    path = Path(path)
    lease = path / "lease.json"
    lease.unlink(missing_ok=True)
    path.rmdir()


def preflight_gpu(max_used_mib=512, max_utilization=10, lease_root=LEASE_ROOT,
                  allowed_indices=None, claim_root=MEMORY_CLAIM_ROOT):
    state = query_gpu_state(cache_max_age_seconds=0)
    leased = leased_uuids(lease_root) | set(active_memory_claims(claim_root))
    selected = select_gpu(
        state, leased, max_used_mib, max_utilization,
        allowed_indices=allowed_indices,
    )
    return {
        "status": "ready" if selected else "blocked_no_free_gpu",
        "selected": selected, "gpus": state, "leased_uuids": sorted(leased),
        "fuser_available": shutil.which("fuser") is not None,
        "policy": {
            "max_used_mib": max_used_mib, "max_utilization": max_utilization,
            "nvitop_is_benign": True, "unknown_compute_blocks": True,
            "foreign_compute_allowed": False,
        },
    }


def run_with_dynamic_gpu(argv, purpose, max_used_mib=512, max_utilization=10,
                         lease_root=LEASE_ROOT, cwd=None, extra_env=None,
                         before_launch=None, required_hostname=None,
                         allowed_gpu_indices=None):
    if required_hostname is not None and socket.gethostname() != str(required_hostname):
        raise RuntimeError(
            f"GPU execution requires host {required_hostname}, got {socket.gethostname()}"
        )
    lease_root = Path(lease_root); lease_root.mkdir(parents=True, exist_ok=True)
    lock_path = PREFLIGHT_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            report = preflight_gpu(
                max_used_mib, max_utilization, lease_root,
                allowed_indices=allowed_gpu_indices,
            )
            if report["status"] != "ready":
                raise RuntimeError("no GPU passed the dynamic preflight")
            gpu = report["selected"]
            lease_path, lease_receipt = acquire_gpu_lease(
                gpu["uuid"], purpose, lease_root,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    environment.update(extra_env or {})
    try:
        if before_launch is not None:
            before_launch()
        completed = subprocess.run(list(argv), cwd=cwd, env=environment, check=False)
        return {"returncode": completed.returncode, "gpu": gpu, "lease": lease_receipt}
    finally:
        release_gpu_lease(lease_path)
