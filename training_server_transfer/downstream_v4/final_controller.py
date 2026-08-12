"""Durable grouped GPU execution for the frozen final downstream protocol."""

import hashlib
import fcntl
import json
import os
import socket
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .commands import PYTHON, build_encode_command, validate_model_receipt
from .baselines import build_kmer_cache
from .data import sha256_path
from .gpu_gate import (
    _host_memory_capacity_report, active_memory_claims,
    run_with_dynamic_gpu, run_with_memory_packed_gpu,
)
from .formal_gate import valid_formal_execution_authorization
from .model_adapters import runtime_spec
from .model_smoke import (
    smoke_contract_sha256, terminal_smoke_failure_path,
    valid_smoke_receipt, valid_terminal_smoke_failure,
)
from .probe import PROBE_PROTOCOL_ID, evaluate_embedding_cache
from .token_probe import TOKEN_PROBE_PROTOCOL_ID
from .zero_shot import ZERO_SHOT_PROTOCOL_ID


CHECKPOINT_ROOT = Path(
    "training_server_transfer/runs/"
    "Stage_B_continuation_3gpu_no_replacement_from_step14000/checkpoints"
)
AUTHORIZED_GPU_INDICES = tuple(range(7))


def _canonical_sha(payload):
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _mode(row):
    if row["task_id"] == "B13":
        return "token"
    if row["task_kind"].startswith("zero_shot"):
        return "zero_shot"
    return "pooled"


def build_gpu_groups(plan, max_tasks_per_group=8):
    buckets = defaultdict(list)
    for row in plan["rows"]:
        if row["execution_kind"] != "evaluation" or row["model_id"] == "kmer_logistic":
            continue
        scope_key = (
            f"step{int(row['checkpoint_step']):05d}"
            if row["checkpoint_scope"] == "step" else "shared"
        )
        key = (scope_key, row["model_id"], int(row["context_bp"]), _mode(row))
        buckets[key].append(row)
    groups = []
    for (scope_key, model_id, context_bp, mode), rows in sorted(buckets.items()):
        rows = sorted(rows, key=lambda row: (row["task_id"], row["run_key"]))
        for part, start in enumerate(range(0, len(rows), int(max_tasks_per_group))):
            selected = rows[start:start + int(max_tasks_per_group)]
            group_key = (
                f"{scope_key}__{model_id}__ctx{context_bp}__{mode}__part{part:02d}"
            )
            groups.append({
                "group_key": group_key,
                "scope_key": scope_key,
                "checkpoint_step": selected[0].get("checkpoint_step"),
                "model_id": model_id,
                "model_kind": selected[0]["model_kind"],
                "context_bp": context_bp,
                "mode": mode,
                "task_ids": [row["task_id"] for row in selected],
                "primary_metrics": [row["primary_metric"] for row in selected],
                "run_keys": [row["run_key"] for row in selected],
            })
    return sorted(groups, key=lambda row: row["group_key"])


def _checkpoint_for_group(group, project_root, final_root):
    model_id = group["model_id"]
    if model_id == "CropGenomeFM":
        step = int(group["checkpoint_step"])
        try:
            plan = json.loads((Path(final_root) / "FINAL_PROTOCOL.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            plan = {}
        identity = (plan.get("checkpoint_identities") or {}).get(f"step_{step:08d}")
        if identity and identity.get("path"):
            checkpoint = Path(identity["path"])
            if not checkpoint.is_absolute():
                checkpoint = Path(project_root) / checkpoint
            return checkpoint.resolve()
        return project_root / CHECKPOINT_ROOT / f"step_{step:08d}.pt"
    return None


def _model_config_for_group(group, project_root, final_root):
    del group
    try:
        plan = json.loads((Path(final_root) / "FINAL_PROTOCOL.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        plan = {}
    identity = plan.get("project_model_config") or {}
    model_config = Path(
        identity.get("path") or "training_server_transfer/configs/model_large.json"
    )
    if not model_config.is_absolute():
        model_config = Path(project_root) / model_config
    return model_config.resolve()


def _checkpoint_record(checkpoint, final_root):
    checkpoint = Path(checkpoint).resolve(); final_root = Path(final_root).resolve()
    inventory_path = final_root / "CHECKPOINT_INVENTORY.json"
    lock_path = final_root / "controller" / "checkpoint_inventory.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            try:
                inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                inventory = {"version": 1, "immutable_checkpoints": True, "records": {}}
            records = inventory.setdefault("records", {})
            before = checkpoint.stat(); key = str(checkpoint)
            record = records.get(key)
            if record and (
                int(record.get("size_bytes", -1)) == before.st_size
                and int(record.get("mtime_ns", -1)) == before.st_mtime_ns
            ):
                return record
            digest = sha256_path(checkpoint)
            after = checkpoint.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"checkpoint changed while hashing: {checkpoint}")
            record = {
                "path": key, "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
                "sha256": digest,
                "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            records[key] = record; inventory["updated_at"] = record["verified_at"]
            _write_json(inventory_path, inventory)
            return record
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _group_paths(group, final_root):
    group_root = Path(final_root) / "gpu_groups" / group["group_key"]
    return {
        "root": group_root,
        "receipt": group_root / "GROUP_RECEIPT.json",
        "failure": group_root / "FAILED.json",
        "log": group_root / "worker.log",
    }


def _group_namespace(group):
    return (
        str(group["scope_key"]), str(group["model_id"]),
        int(group["context_bp"]), str(group["mode"]),
    )


def _namespace_lock_path(group, final_root):
    encoded = "|".join(map(str, _group_namespace(group))).encode("utf-8")
    return (
        Path(final_root) / "controller" / "namespace_locks"
        / (hashlib.sha256(encoded).hexdigest() + ".lock")
    )


def _assert_current_frozen_plan(final_root, expected_plan_sha256):
    plan_path = Path(final_root) / "FINAL_PROTOCOL.json"
    try:
        current = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "formal execution authorization revoked: current protocol is unreadable"
        ) from error
    if (
        current.get("plan_sha256") != expected_plan_sha256
        or current.get("checkpoint_identity_state") != "frozen_complete"
    ):
        raise RuntimeError(
            "formal execution authorization revoked: current protocol is not the worker's frozen plan"
        )
    if valid_formal_execution_authorization(final_root, expected_plan_sha256) is None:
        raise RuntimeError("formal execution authorization missing or stale")
    return current


def _daemon_status(group_keys, complete, terminal_failures):
    if len(complete) == len(group_keys):
        return "ok"
    if len(set(complete) | set(terminal_failures)) == len(group_keys):
        return "failed"
    return "running"


def _resolve_daemon_worker_limit(max_workers, plan,
                                 allowed_gpu_indices=AUTHORIZED_GPU_INDICES):
    configured = int(max_workers)
    if configured < 0:
        raise ValueError("max_workers must be zero (use frozen plan) or positive")
    policy = plan.get("gpu_scheduler_policy") or {}
    frozen = int(policy.get("max_workers", min(3, len(tuple(allowed_gpu_indices)))))
    if not 1 <= frozen <= 3:
        raise RuntimeError("frozen max_workers must be between 1 and 3")
    if configured > frozen:
        raise RuntimeError(
            f"runtime max_workers={configured} exceeds frozen max_workers={frozen}"
        )
    return configured or frozen


def _claimed_group_keys(claims_by_uuid):
    prefix = "downstream-final:"
    keys = set()
    for claims in (claims_by_uuid or {}).values():
        for claim in claims:
            purpose = str(claim.get("purpose", ""))
            if purpose.startswith(prefix):
                keys.add(purpose[len(prefix):].split(":token-probe", 1)[0])
    return keys


def _daemon_launch_slots(worker_limit, active_group_keys, claimed_group_keys,
                         max_unclaimed_workers=1):
    active_keys = set(active_group_keys)
    if worker_limit is not None:
        return max(0, int(worker_limit) - len(active_keys))
    unclaimed = active_keys - set(claimed_group_keys)
    return max(0, int(max_unclaimed_workers) - len(unclaimed))


def _host_memory_budget_for_group(group, gpu_memory_budget_mib):
    if group["model_id"] == "CropGenomeFM":
        return 49152
    return max(8192, 2 * int(gpu_memory_budget_mib))


def _memory_policy_for_group(plan, group, foreign_compute_allowed=None,
                             max_tasks_per_gpu=None):
    policy = plan.get("gpu_scheduler_policy") or {}
    if policy.get("mode") != "memory_packed":
        return None
    key = f"{group['model_id']}:{int(group['context_bp'])}:{group['mode']}"
    budgets = policy.get("budgets_mib") or {}
    if key not in budgets:
        raise RuntimeError(f"missing frozen GPU memory budget: {key}")
    frozen_max_tasks = int(policy.get("max_tasks_per_gpu", 1))
    frozen_foreign = bool(policy.get("foreign_compute_allowed", False))
    runtime_max_tasks = (
        frozen_max_tasks if max_tasks_per_gpu is None else int(max_tasks_per_gpu)
    )
    runtime_foreign = (
        frozen_foreign
        if foreign_compute_allowed is None else bool(foreign_compute_allowed)
    )
    if runtime_max_tasks != frozen_max_tasks or runtime_foreign != frozen_foreign:
        raise RuntimeError(
            "runtime scheduler override relaxes frozen GPU memory policy"
        )
    resolved = {
        "memory_budget_mib": int(budgets[key]),
        "reserved_headroom_mib": int(policy.get("reserved_headroom_mib", 1024)),
        "minimum_runtime_headroom_mib": int(
            policy.get("minimum_runtime_headroom_mib", 512)
        ),
        "max_tasks_per_gpu": frozen_max_tasks,
        "host_memory_budget_mib": _host_memory_budget_for_group(
            group, int(budgets[key]),
        ),
        "reserved_host_headroom_mib": int(
            policy.get("host_reserved_memory_mib", 8192)
        ),
        "wait_timeout_seconds": int(policy.get("wait_timeout_seconds", 7200)),
        "poll_seconds": float(policy.get("poll_seconds", 15)),
        "monitor_seconds": float(policy.get("monitor_seconds", 2)),
        "foreign_compute_allowed": frozen_foreign,
    }
    if (
        resolved["memory_budget_mib"] <= 0
        or resolved["reserved_headroom_mib"] < 0
        or resolved["minimum_runtime_headroom_mib"] < 0
        or not 1 <= resolved["max_tasks_per_gpu"] <= 3
        or resolved["host_memory_budget_mib"] <= 0
        or resolved["reserved_host_headroom_mib"] < 0
        or not 1 <= int(policy.get("max_tasks_per_gpu", 1)) <= 3
        or not isinstance(policy.get("foreign_compute_allowed"), bool)
        or policy.get("unknown_compute_blocks") is not True
        or policy.get("do_not_signal_foreign_processes", True) is not True
        or policy.get("nvitop_is_benign") is not True
    ):
        raise RuntimeError(f"invalid frozen GPU memory policy: {key}")
    return resolved


def _daemon_group_host_capacity(plan, group, claims_by_uuid,
                                available_host_memory_mib=None,
                                process_rss_lookup=None):
    """Advisory host-memory precheck used before a daemon spawns a worker."""
    policy = _memory_policy_for_group(plan, group)
    if policy is None:
        return True, {"status": "ready", "policy": "not_memory_packed"}
    kwargs = {}
    if available_host_memory_mib is not None:
        kwargs["available_host_memory_mib"] = int(available_host_memory_mib)
    if process_rss_lookup is not None:
        kwargs["process_rss_lookup"] = process_rss_lookup
    report = _host_memory_capacity_report(
        claims_by_uuid,
        required_host_memory_mib=policy["host_memory_budget_mib"],
        reserved_host_headroom_mib=policy["reserved_host_headroom_mib"],
        **kwargs,
    )
    return report["status"] == "ready", report


def _run_group_gpu_command(command, purpose, project_root, extra_env,
                           authorization_check, memory_policy):
    if memory_policy is None:
        return run_with_dynamic_gpu(
            command, purpose, cwd=project_root, extra_env=extra_env,
            before_launch=authorization_check, required_hostname="gpu05",
            allowed_gpu_indices=AUTHORIZED_GPU_INDICES,
        )
    return run_with_memory_packed_gpu(
        command, purpose, cwd=project_root, extra_env=extra_env,
        before_launch=authorization_check, required_hostname="gpu05",
        allowed_gpu_indices=AUTHORIZED_GPU_INDICES, **memory_policy,
    )


def _heartbeat_path(final_root):
    return Path(final_root) / "controller" / "GPU_DAEMON_HEARTBEAT.json"


def _heartbeat_payload(plan_sha256, poll_seconds, status="running"):
    return {
        "status": status,
        "plan_sha256": plan_sha256,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "poll_seconds": int(poll_seconds),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _heartbeat_is_live(record, plan_sha256):
    if not isinstance(record, dict) or record.get("plan_sha256") != plan_sha256:
        return False
    try:
        updated = datetime.fromisoformat(record["updated_at"])
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        fresh = age <= max(180, 3 * int(record.get("poll_seconds", 60)))
    except (KeyError, TypeError, ValueError):
        return False
    if not fresh or record.get("status") != "running":
        return False
    if record.get("hostname") != socket.gethostname():
        return True
    try:
        command = (Path("/proc") / str(int(record["pid"])) / "cmdline").read_bytes()
    except (OSError, TypeError, ValueError):
        return False
    return b"run_cropgenome_downstream_final.py" in command and b"daemon" in command


def _daemon_authorization_ready(plan, final_root):
    authorization = valid_formal_execution_authorization(
        final_root, plan["plan_sha256"],
    )
    if authorization is None:
        return False
    for record in authorization.get("evidence") or []:
        path = Path(record.get("path") or "")
        if not path.name.startswith("OPERATIONAL_SCHEDULER_AMENDMENT"):
            continue
        try:
            amendment = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if (
            amendment.get("plan_sha256") != plan["plan_sha256"]
            or amendment.get("new_formal_worker_launch_allowed") is not True
        ):
            return False
    return True


def current_controller_status(plan, final_root):
    """Reconcile immutable receipts, live locks and a plan-bound daemon heartbeat."""
    final_root = Path(final_root).resolve()
    groups = build_gpu_groups(plan)
    by_key = {group["group_key"]: group for group in groups}
    complete = {
        key for key, group in by_key.items()
        if _valid_group_receipt(group, final_root, plan["plan_sha256"])
    }
    terminal = {
        key for key, group in by_key.items() if key not in complete
        and _valid_terminal_group_failure(group, final_root, plan["plan_sha256"])
    }
    active = {
        key for key, group in by_key.items()
        if key not in complete and key not in terminal and _group_lock_is_held(group, final_root)
    }
    try:
        heartbeat = json.loads(_heartbeat_path(final_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        heartbeat = None
    daemon_live = _heartbeat_is_live(heartbeat, plan["plan_sha256"])
    if len(complete) == len(groups):
        status = "ok"
    elif len(complete | terminal) == len(groups):
        status = "failed"
    elif active or daemon_live:
        status = "running"
    else:
        status = "not_running"
    snapshot_path = final_root / "GPU_CONTROLLER_STATUS.json"
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        snapshot = None
    return {
        "status": status,
        "protocol_status": plan.get("checkpoint_identity_state"),
        "plan_sha256": plan["plan_sha256"],
        "groups_total": len(groups),
        "groups_complete": len(complete),
        "groups_active": sorted(active),
        "terminal_failures": sorted(terminal),
        "daemon_live": daemon_live,
        "heartbeat": heartbeat,
        "stale_snapshot_detected": bool(
            snapshot and snapshot.get("status") == "running" and not daemon_live and not active
        ),
        "snapshot_updated_at": snapshot.get("updated_at") if snapshot else None,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _group_lock_is_held(group, final_root):
    """Return whether a worker from this or an earlier daemon owns the group."""
    paths = _group_paths(group, final_root)
    lock_path = paths["root"] / ".run.lock"
    if not lock_path.is_file():
        return False
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def _valid_group_receipt(group, final_root, plan_sha256):
    path = _group_paths(group, final_root)["receipt"]
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    canonical = dict(receipt); stored = canonical.pop("receipt_sha256", None)
    if (
        receipt.get("status") != "ok"
        or receipt.get("group_key") != group["group_key"]
        or receipt.get("plan_sha256") != plan_sha256
        or stored != _canonical_sha(canonical)
    ):
        return None
    try:
        expected_paths = {record["path"] for record in _group_artifacts(group, final_root)}
    except RuntimeError:
        return None
    recorded_paths = {record.get("path") for record in receipt.get("artifacts") or []}
    if recorded_paths != expected_paths:
        return None
    for record in receipt.get("artifacts") or []:
        artifact = Path(record["path"])
        if not artifact.is_file() or sha256_path(artifact) != record["sha256"]:
            return None
    return receipt


def _valid_terminal_group_failure(group, final_root, plan_sha256):
    path = _group_paths(group, final_root)["failure"]
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    canonical = dict(receipt); stored = canonical.pop("receipt_sha256", None)
    if (
        receipt.get("status") != "terminal_failed"
        or receipt.get("group_key") != group["group_key"]
        or receipt.get("plan_sha256") != plan_sha256
        or stored != _canonical_sha(canonical)
    ):
        return None
    for record in receipt.get("artifacts") or []:
        artifact = Path(record["path"])
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(record["size_bytes"])
            or sha256_path(artifact) != record["sha256"]
        ):
            return None
    return receipt


def _write_terminal_group_failure(group, final_root, plan_sha256, attempts, reason,
                                  dependency=None):
    paths = _group_paths(group, final_root)
    artifacts = []
    for artifact in [paths["log"], Path(dependency) if dependency else None]:
        if artifact is not None and artifact.is_file():
            artifacts.append({
                "path": str(artifact), "sha256": sha256_path(artifact),
                "size_bytes": artifact.stat().st_size,
            })
    receipt = {
        "status": "terminal_failed", "group_key": group["group_key"],
        "plan_sha256": plan_sha256, "group": group,
        "attempts": int(attempts), "reason": str(reason),
        "dependency": str(dependency) if dependency else None,
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    _write_json(paths["failure"], receipt)
    return receipt


def group_readiness(group, registry, project_root, final_root):
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    missing_datasets = [
        task_id for task_id in group["task_ids"]
        if not (final_root / "datasets" / task_id / "samples.tsv").is_file()
    ]
    if missing_datasets:
        return False, f"datasets:{','.join(missing_datasets)}"
    checkpoint = _checkpoint_for_group(group, project_root, final_root)
    if checkpoint is not None and not checkpoint.is_file():
        return False, f"checkpoint:{checkpoint}"
    if checkpoint is not None:
        try:
            plan = json.loads((final_root / "FINAL_PROTOCOL.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            plan = {}
        identity_key = f"step_{int(group['checkpoint_step']):08d}"
        identity = (plan.get("checkpoint_identities") or {}).get(identity_key)
        if identity:
            stat = checkpoint.stat()
            if (
                stat.st_size != int(identity.get("size_bytes", -1))
                or stat.st_mtime_ns != int(identity.get("mtime_ns", -1))
            ):
                return False, f"checkpoint_identity:{identity_key}"
        model_config = _model_config_for_group(group, project_root, final_root)
        config_identity = plan.get("project_model_config") or {}
        if not model_config.is_file():
            return False, f"model_config:{model_config}"
        if config_identity:
            stat = model_config.stat()
            if (
                stat.st_size != int(config_identity.get("size_bytes", -1))
                or stat.st_mtime_ns != int(config_identity.get("mtime_ns", -1))
                or sha256_path(model_config) != config_identity.get("sha256")
            ):
                return False, "model_config_identity"
    if group["model_kind"] == "public_weight":
        receipt = (
            project_root / "training_server_transfer/public_models/downstream_v4"
            / group["model_id"] / "download_receipt.json"
        )
        if not receipt.is_file():
            return False, f"model:{group['model_id']}"
        model = next(row for row in registry["models"] if row["model_id"] == group["model_id"])
        contract = smoke_contract_sha256(model, project_root)
        terminal = valid_terminal_smoke_failure(final_root, group["model_id"], contract)
        if terminal:
            return False, f"model_smoke_terminal:{terminal_smoke_failure_path(final_root, group['model_id'])}"
        if not valid_smoke_receipt(final_root, group["model_id"], contract):
            return False, f"model_smoke:{group['model_id']}"
    return True, "ready"


def _replace_repeated_option(argv, option, values):
    argv = list(argv)
    while option in argv:
        index = argv.index(option)
        del argv[index:index + 2]
    for value in values:
        argv.extend([option, str(value)])
    return argv


def _build_group_command(group, registry, project_root, final_root, script_path):
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    checkpoint = _checkpoint_for_group(group, project_root, final_root)
    model_config = _model_config_for_group(group, project_root, final_root)
    cache_base = final_root / "caches" / group["mode"] / group["scope_key"]
    if group["mode"] == "zero_shot":
        spec_path = _group_paths(group, final_root)["root"] / "ZERO_SHOT_SPEC.json"
        spec = {
            **group, "project_root": str(project_root), "final_root": str(final_root),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "model_config": str(model_config),
        }
        _write_json(spec_path, spec)
        return [PYTHON, str(Path(script_path).resolve()), "zero-child", "--spec", str(spec_path)]
    first_task = group["task_ids"][0]
    command = build_encode_command(
        registry, first_task, group["model_id"], project_root,
        final_root / "datasets" / first_task, cache_base,
        checkpoint=checkpoint, model_config=model_config,
        context=group["context_bp"], device="cuda",
    )
    if group["mode"] == "pooled":
        command = _replace_repeated_option(command, "--task", group["task_ids"])
    return command


def _group_artifacts(group, final_root):
    final_root = Path(final_root).resolve(); artifacts = []
    if group["mode"] == "zero_shot":
        candidates = []
        for run_key, expected_primary in zip(group["run_keys"], group["primary_metrics"]):
            receipt_path = final_root / "zero_shot" / run_key / "FINAL_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
            canonical = dict(receipt); stored = canonical.pop("receipt_sha256", None)
            prediction = Path(receipt.get("predictions", ""))
            bootstrap = receipt.get("bootstrap") or {}
            bootstrap_source = Path(bootstrap.get("source_data", ""))
            if (
                receipt.get("status") != "ok"
                or receipt.get("evaluation_protocol_id") != ZERO_SHOT_PROTOCOL_ID
                or receipt.get("test_access_count") != 1
                or receipt.get("primary_metric") != expected_primary
                or receipt.get("primary_metric") != bootstrap.get("primary_metric")
                or int(bootstrap.get("replicates", -1)) != 1000
                or stored != _canonical_sha(canonical)
                or not prediction.is_file()
                or sha256_path(prediction) != receipt.get("predictions_sha256")
                or not bootstrap_source.is_file()
                or sha256_path(bootstrap_source) != bootstrap.get("source_data_sha256")
            ):
                raise RuntimeError(f"invalid zero-shot receipt: {receipt_path}")
            candidates.extend([receipt_path, prediction, bootstrap_source])
    elif group["mode"] == "token":
        cache_manifest = (
            final_root / "caches" / "token" / group["scope_key"]
            / group["model_id"] / f"context_{group['context_bp']}" / "cache_manifest.json"
        )
        final_path = final_root / "results" / group["run_keys"][0] / "FINAL_RECEIPT.json"
        final_receipt = json.loads(final_path.read_text(encoding="utf-8")) if final_path.is_file() else {}
        if (
            len(final_receipt.get("seed_test_metrics") or []) != 5
            or final_receipt.get("probe_protocol_id") != TOKEN_PROBE_PROTOCOL_ID
        ):
            raise RuntimeError("token result lacks the frozen five independent seed metrics")
        prediction = Path(final_receipt.get("test_predictions", ""))
        candidates = [cache_manifest, final_path, prediction]
    else:
        candidates = [
            final_root / "caches" / "pooled" / group["scope_key"]
            / group["model_id"] / f"context_{group['context_bp']}" / f"{task_id}.npz"
            for task_id in group["task_ids"]
        ]
    for path in candidates:
        if not path.is_file():
            raise RuntimeError(f"group output is missing: {path}")
        artifacts.append({"path": str(path), "sha256": sha256_path(path), "size_bytes": path.stat().st_size})
    return artifacts


def _execute_group_unlocked(group, registry, project_root, final_root, plan_sha256,
                            script_path, foreign_compute_allowed=False,
                            max_tasks_per_gpu=1):
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    paths = _group_paths(group, final_root); paths["root"].mkdir(parents=True, exist_ok=True)
    resumed = _valid_group_receipt(group, final_root, plan_sha256)
    if resumed:
        return {**resumed, "resumed": True}
    ready, reason = group_readiness(group, registry, project_root, final_root)
    if not ready:
        raise RuntimeError(f"group is not ready: {reason}")
    command = _build_group_command(group, registry, project_root, final_root, script_path)
    model = next(row for row in registry["models"] if row["model_id"] == group["model_id"])
    extra_env = {}
    compiler_dir = Path(PYTHON).resolve().parent
    cc = compiler_dir / "x86_64-conda-linux-gnu-cc"
    cxx = compiler_dir / "x86_64-conda-linux-gnu-c++"
    if cc.is_file():
        extra_env["CC"] = str(cc)
    if cxx.is_file():
        extra_env["CXX"] = str(cxx)
    if model.get("runtime_pythonpath"):
        runtime = project_root / model["runtime_pythonpath"]
        inherited = os.environ.get("PYTHONPATH")
        extra_env["PYTHONPATH"] = str(runtime) + (os.pathsep + inherited if inherited else "")
    started = time.time()
    current_plan = _assert_current_frozen_plan(final_root, plan_sha256)
    memory_policy = _memory_policy_for_group(
        current_plan, group, foreign_compute_allowed=foreign_compute_allowed,
        max_tasks_per_gpu=max_tasks_per_gpu,
    )
    authorization_check = lambda: _assert_current_frozen_plan(final_root, plan_sha256)
    try:
        execution = _run_group_gpu_command(
            command, f"downstream-final:{group['group_key']}", project_root,
            extra_env, authorization_check, memory_policy,
        )
        if int(execution["returncode"]):
            raise RuntimeError(f"group command exited {execution['returncode']}")
        post_execution = None
        if group["mode"] == "token":
            cache_dir = (
                final_root / "caches" / "token" / group["scope_key"]
                / group["model_id"] / f"context_{group['context_bp']}"
            )
            output_root = final_root / "results" / group["run_keys"][0]
            token_command = [
                PYTHON, str(project_root / "scripts/run_cropgenome_downstream_v4.py"),
                "--project-root", str(project_root), "token-probe",
                "--cache-dir", str(cache_dir), "--output-root", str(output_root),
                "--device", "cuda",
            ]
            post_execution = _run_group_gpu_command(
                token_command, f"downstream-final:{group['group_key']}:token-probe",
                project_root, {}, authorization_check, memory_policy,
            )
            if int(post_execution["returncode"]):
                raise RuntimeError(f"token probe exited {post_execution['returncode']}")
        artifacts = _group_artifacts(group, final_root)
        checkpoint = _checkpoint_for_group(group, project_root, final_root)
        receipt = {
            "status": "ok", "group_key": group["group_key"],
            "plan_sha256": plan_sha256, "group": group, "argv": command,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": (
                _checkpoint_record(checkpoint, final_root)["sha256"] if checkpoint else None
            ),
            "execution": execution, "post_execution": post_execution,
            "artifacts": artifacts,
            "elapsed_seconds": time.time() - started,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "resumed": False,
        }
        receipt["receipt_sha256"] = _canonical_sha(receipt)
        _write_json(paths["receipt"], receipt)
        paths["failure"].unlink(missing_ok=True)
        return receipt
    except Exception as error:
        failure = {
            "status": "failed", "group_key": group["group_key"], "error": repr(error),
            "elapsed_seconds": time.time() - started,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _write_json(paths["failure"], failure)
        raise


def execute_group(group, registry, project_root, final_root, plan_sha256, script_path,
                  foreign_compute_allowed=None, max_tasks_per_gpu=None):
    paths = _group_paths(group, final_root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    lock_path = paths["root"] / ".run.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            namespace_path = _namespace_lock_path(group, final_root)
            namespace_path.parent.mkdir(parents=True, exist_ok=True)
            with namespace_path.open("a+") as namespace_handle:
                fcntl.flock(namespace_handle.fileno(), fcntl.LOCK_EX)
                try:
                    _assert_current_frozen_plan(final_root, plan_sha256)
                    return _execute_group_unlocked(
                        group, registry, project_root, final_root, plan_sha256, script_path,
                        foreign_compute_allowed=foreign_compute_allowed,
                        max_tasks_per_gpu=max_tasks_per_gpu,
                    )
                finally:
                    fcntl.flock(namespace_handle.fileno(), fcntl.LOCK_UN)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def run_zero_child(spec_path, registry):
    from .zero_shot import CropGenomeMLMScorer, HuggingFaceMLMScorer, run_zero_shot_task
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    project_root = Path(spec["project_root"]); final_root = Path(spec["final_root"])
    model = next(row for row in registry["models"] if row["model_id"] == spec["model_id"])
    context = int(spec["context_bp"])
    if model["kind"] == "public_weight":
        model_root = project_root / "training_server_transfer/public_models/downstream_v4" / spec["model_id"]
        validate_model_receipt(model, model_root)
        runtime = runtime_spec(spec["model_id"])
        if context > int(runtime["context_bp"]):
            raise RuntimeError("zero-shot context exceeds public model runtime limit")
        scorer = HuggingFaceMLMScorer(
            model_root, context, "cuda",
            inference_dtype=runtime.get("inference_dtype", "float16"),
            load_dtype=runtime.get("load_dtype"),
        )
        batch_size = int(runtime["batch_size"])
    else:
        scorer = CropGenomeMLMScorer(
            project_root, spec["checkpoint"], spec["model_config"], context, "cuda",
        )
        scorer.model_id = spec["model_id"]
        batch_size = 1
    results = []
    for task_id, run_key, primary_metric in zip(
        spec["task_ids"], spec["run_keys"], spec["primary_metrics"],
    ):
        output = final_root / "zero_shot" / run_key
        receipt = run_zero_shot_task(
            task_id, final_root / "datasets" / task_id, scorer, output, batch_size,
            primary_metric=primary_metric,
        )
        results.append({"task_id": task_id, "run_key": run_key, "receipt": receipt})
    return {"status": "ok", "model_id": spec["model_id"], "context_bp": context, "results": results}


def _row_scope_key(row):
    if row["checkpoint_scope"] == "step":
        return f"step{int(row['checkpoint_step']):05d}"
    return "shared"


def _cpu_row_receipt_path(row, final_root):
    return Path(final_root) / "results" / row["run_key"] / "ROW_RECEIPT.json"


def _valid_cpu_row_receipt(row, final_root, plan_sha256):
    path = _cpu_row_receipt_path(row, final_root)
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    canonical = dict(receipt); stored = canonical.pop("receipt_sha256", None)
    if (
        receipt.get("status") != "ok" or receipt.get("run_key") != row["run_key"]
        or receipt.get("plan_sha256") != plan_sha256
        or stored != _canonical_sha(canonical)
    ):
        return None
    for record in receipt.get("artifacts") or []:
        path = Path(record["path"])
        if not path.is_file() or sha256_path(path) != record["sha256"]:
            return None
    result_path = Path(receipt.get("result_receipt", ""))
    if not result_path.is_file():
        return None
    try:
        result_receipt = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    expected_probe_protocol = (
        TOKEN_PROBE_PROTOCOL_ID if row["task_id"] == "B13" else PROBE_PROTOCOL_ID
    )
    if (
        len(result_receipt.get("seed_test_metrics") or []) != 5
        or result_receipt.get("probe_protocol_id") != expected_probe_protocol
    ):
        return None
    return receipt


def _cpu_terminal_failure_path(row, final_root):
    return Path(final_root) / "results" / row["run_key"] / "TERMINAL_FAILURE.json"


def _valid_terminal_cpu_failure(row, final_root, plan_sha256):
    path = _cpu_terminal_failure_path(row, final_root)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    canonical = dict(receipt); stored = canonical.pop("receipt_sha256", None)
    if (
        receipt.get("status") != "terminal_failed"
        or receipt.get("run_key") != row["run_key"]
        or receipt.get("plan_sha256") != plan_sha256
        or stored != _canonical_sha(canonical)
    ):
        return None
    for record in receipt.get("artifacts") or []:
        artifact = Path(record["path"])
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(record["size_bytes"])
            or sha256_path(artifact) != record["sha256"]
        ):
            return None
    return receipt


def record_cpu_row_failure(row, final_root, plan_sha256, error_text, max_attempts=3):
    result_root = Path(final_root) / "results" / row["run_key"]
    attempts_root = result_root / "attempt_failures" / str(plan_sha256)
    attempts_root.mkdir(parents=True, exist_ok=True)
    with (result_root / ".failure.lock").open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            terminal = _valid_terminal_cpu_failure(row, final_root, plan_sha256)
            if terminal:
                return terminal
            matching = []
            for path in sorted(attempts_root.glob("attempt_*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                canonical = dict(payload); stored = canonical.pop("receipt_sha256", None)
                if (
                    payload.get("status") == "failed"
                    and payload.get("run_key") == row["run_key"]
                    and payload.get("plan_sha256") == plan_sha256
                    and stored == _canonical_sha(canonical)
                ):
                    matching.append((path, payload))
            attempt = len(matching) + 1
            failure = {
                "status": "failed", "run_key": row["run_key"],
                "plan_sha256": plan_sha256, "attempt": attempt,
                "row": row, "error": str(error_text),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            failure["receipt_sha256"] = _canonical_sha(failure)
            failure_path = attempts_root / f"attempt_{attempt:02d}.json"
            _write_json(failure_path, failure)
            matching.append((failure_path, failure))
            if attempt < int(max_attempts):
                return failure
            artifacts = [
                {"path": str(path), "sha256": sha256_path(path),
                 "size_bytes": path.stat().st_size}
                for path, _ in matching
            ]
            terminal = {
                "status": "terminal_failed", "run_key": row["run_key"],
                "plan_sha256": plan_sha256, "row": row,
                "attempts": attempt, "reason": "cpu_row_failed_max_attempts",
                "artifacts": artifacts,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            terminal["receipt_sha256"] = _canonical_sha(terminal)
            _write_json(_cpu_terminal_failure_path(row, final_root), terminal)
            return terminal
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def cpu_row_readiness(row, final_root):
    final_root = Path(final_root).resolve()
    if row["execution_kind"] != "evaluation":
        return False, "not_evaluation"
    if row["task_id"] == "B13" or row["task_kind"].startswith("zero_shot"):
        return False, "gpu_closed"
    dataset = final_root / "datasets" / row["task_id"] / "samples.tsv"
    if not dataset.is_file():
        return False, "dataset"
    if row["model_id"] == "kmer_logistic":
        return True, "ready"
    cache = (
        final_root / "caches" / "pooled" / _row_scope_key(row)
        / row["model_id"] / f"context_{int(row['context_bp'])}" / f"{row['task_id']}.npz"
    )
    return (True, "ready") if cache.is_file() else (False, "cache")


def _execute_cpu_row_unlocked(row, project_root, final_root, plan_sha256,
                              seeds=None):
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    resumed = _valid_cpu_row_receipt(row, final_root, plan_sha256)
    if resumed:
        return {**resumed, "resumed": True}
    ready, reason = cpu_row_readiness(row, final_root)
    if not ready:
        raise RuntimeError(f"CPU row is not ready: {reason}")
    dataset_root = final_root / "datasets" / row["task_id"]
    result_root = final_root / "results" / row["run_key"]
    if row["model_id"] == "kmer_logistic":
        cache = (
            final_root / "caches" / "kmer" / "shared"
            / f"context_{int(row['context_bp'])}" / f"{row['task_id']}.npz"
        )
        baseline = build_kmer_cache(
            dataset_root, cache, row["task_id"], context=int(row["context_bp"]),
        )
    else:
        cache = (
            final_root / "caches" / "pooled" / _row_scope_key(row)
            / row["model_id"] / f"context_{int(row['context_bp'])}" / f"{row['task_id']}.npz"
        )
        baseline = None
    cv_policy = row["cv_policy"]
    resolved_seeds = tuple(
        int(seed) for seed in (
            seeds if seeds is not None else row.get("seeds") or (13, 29, 43, 71, 97)
        )
    )
    result = evaluate_embedding_cache(
        cache, row["task_kind"], result_root,
        seeds=resolved_seeds, cv_policy=cv_policy,
        few_shot_regimes=tuple(row.get("few_shot_regimes") or ()),
        training_row_cap=row.get("training_row_cap"),
    )
    result_receipt = result_root / "FINAL_RECEIPT.json"
    prediction_path = Path(result["test_predictions"])
    artifacts = [
        {"path": str(cache), "sha256": sha256_path(cache), "size_bytes": cache.stat().st_size},
        {"path": str(result_receipt), "sha256": sha256_path(result_receipt), "size_bytes": result_receipt.stat().st_size},
        {"path": str(prediction_path), "sha256": sha256_path(prediction_path), "size_bytes": prediction_path.stat().st_size},
    ]
    receipt = {
        "status": "ok", "run_key": row["run_key"], "plan_sha256": plan_sha256,
        "row": row, "seeds": list(resolved_seeds),
        "test_metrics": result["test_metrics"],
        "few_shot_results": result.get("few_shot_results") or [],
        "cache": str(cache), "result_receipt": str(result_receipt),
        "baseline_receipt": baseline, "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resumed": False,
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    _write_json(_cpu_row_receipt_path(row, final_root), receipt)
    return receipt


def execute_cpu_row(row, project_root, final_root, plan_sha256,
                    seeds=None):
    receipt_path = _cpu_row_receipt_path(row, final_root)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with (receipt_path.parent / ".run.lock").open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _execute_cpu_row_unlocked(
                row, project_root, final_root, plan_sha256, seeds=seeds,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _group_worker_command(script_path, group_key, final_root,
                          foreign_compute_allowed=False, max_tasks_per_gpu=1):
    command = [
        PYTHON, str(Path(script_path).resolve()), "run-group",
        "--group-key", str(group_key), "--final-root", str(final_root),
    ]
    if foreign_compute_allowed:
        command.append("--allow-foreign-compute")
    command.extend(["--max-tasks-per-gpu", str(int(max_tasks_per_gpu))])
    return command


def run_daemon(plan, registry, project_root, final_root, script_path, max_workers=0,
               poll_seconds=60, max_attempts=3, foreign_compute_allowed=None,
               max_tasks_per_gpu=None):
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    frozen_policy = plan.get("gpu_scheduler_policy") or {}
    frozen_foreign = bool(frozen_policy.get("foreign_compute_allowed", False))
    frozen_max_tasks = int(frozen_policy.get("max_tasks_per_gpu", 1))
    if foreign_compute_allowed is None:
        foreign_compute_allowed = frozen_foreign
    if max_tasks_per_gpu is None:
        max_tasks_per_gpu = frozen_max_tasks
    if (
        bool(foreign_compute_allowed) != frozen_foreign
        or int(max_tasks_per_gpu) != frozen_max_tasks
    ):
        raise RuntimeError("runtime scheduler override relaxes frozen GPU memory policy")
    groups = build_gpu_groups(plan); by_key = {group["group_key"]: group for group in groups}
    state_path = final_root / "GPU_CONTROLLER_STATUS.json"
    attempts_path = final_root / "controller" / "GPU_ATTEMPTS.json"
    try:
        attempt_state = json.loads(attempts_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        attempt_state = {}
    if attempt_state.get("plan_sha256") != plan["plan_sha256"]:
        attempt_state = {"plan_sha256": plan["plan_sha256"], "attempts": {}}
    attempts = defaultdict(int, {
        key: int(value) for key, value in attempt_state.get("attempts", {}).items()
    })
    public_models = {
        row["model_id"]: row for row in registry["models"] if row["kind"] == "public_weight"
    }
    smoke_contracts = {
        model_id: smoke_contract_sha256(model, project_root)
        for model_id, model in public_models.items()
    }
    worker_limit = _resolve_daemon_worker_limit(
        max_workers, plan, AUTHORIZED_GPU_INDICES,
    )
    worker_limit_source = "configured" if int(max_workers) else "frozen_policy"
    active = {}
    while True:
        _write_json(
            _heartbeat_path(final_root),
            _heartbeat_payload(plan["plan_sha256"], poll_seconds),
        )
        for key, item in list(active.items()):
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["log_handle"].close(); del active[key]
            if returncode:
                attempts[key] += 1
                attempt_state["attempts"][key] = attempts[key]
                _write_json(attempts_path, attempt_state)
        complete = {
            key for key, group in by_key.items()
            if _valid_group_receipt(group, final_root, plan["plan_sha256"])
        }
        for key, group in by_key.items():
            if key in complete or key in active:
                continue
            if group["model_id"] in public_models:
                dependency = terminal_smoke_failure_path(final_root, group["model_id"])
                if valid_terminal_smoke_failure(
                    final_root, group["model_id"], smoke_contracts[group["model_id"]],
                ) and not _valid_terminal_group_failure(group, final_root, plan["plan_sha256"]):
                    _write_terminal_group_failure(
                        group, final_root, plan["plan_sha256"], attempts[key],
                        "blocked_by_terminal_model_smoke", dependency=dependency,
                    )
            if (
                attempts[key] >= int(max_attempts)
                and not _valid_terminal_group_failure(group, final_root, plan["plan_sha256"])
            ):
                _write_terminal_group_failure(
                    group, final_root, plan["plan_sha256"], attempts[key],
                    "group_execution_failed_max_attempts",
                )
        terminal_failures = {
            key for key, group in by_key.items()
            if key not in complete
            and _valid_terminal_group_failure(group, final_root, plan["plan_sha256"])
        }
        external_active = {
            key for key, group in by_key.items()
            if key not in active and key not in complete and key not in terminal_failures
            and _group_lock_is_held(group, final_root)
        }
        all_active_keys = set(active) | external_active
        claims_by_uuid = active_memory_claims()
        claimed_group_keys = _claimed_group_keys(claims_by_uuid)
        slots = _daemon_launch_slots(
            worker_limit, all_active_keys, claimed_group_keys,
        )
        active_namespaces = {
            _group_namespace(by_key[key]) for key in set(active) | external_active
        }
        waiting_reasons = defaultdict(int)
        authorization_ready = _daemon_authorization_ready(plan, final_root)
        if slots and authorization_ready:
            for key, group in sorted(by_key.items()):
                if not slots:
                    break
                if (
                    key in complete or key in terminal_failures
                    or key in active or key in external_active
                ):
                    continue
                namespace = _group_namespace(group)
                if namespace in active_namespaces:
                    waiting_reasons["namespace_active"] += 1
                    continue
                ready, reason = group_readiness(group, registry, project_root, final_root)
                if not ready:
                    waiting_reasons[reason.split(":", 1)[0]] += 1; continue
                host_ready, host_report = _daemon_group_host_capacity(
                    plan, group, active_memory_claims(),
                )
                if not host_ready:
                    waiting_reasons[host_report["status"]] += 1
                    continue
                paths = _group_paths(group, final_root); paths["root"].mkdir(parents=True, exist_ok=True)
                log_handle = paths["log"].open("ab", buffering=0)
                command = _group_worker_command(
                    script_path, key, final_root, foreign_compute_allowed,
                    max_tasks_per_gpu,
                )
                process = subprocess.Popen(
                    command, cwd=project_root, stdout=log_handle,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                active[key] = {"process": process, "log_handle": log_handle, "pid": process.pid}
                active_namespaces.add(namespace)
                slots -= 1; time.sleep(3)
        elif slots:
            waiting_reasons["formal_execution_authorization"] = 1
        status = {
            "status": _daemon_status(by_key, complete, terminal_failures),
            "groups_total": len(groups), "groups_complete": len(complete),
            "groups_active": {key: item["pid"] for key, item in active.items()},
            "groups_external_active": sorted(external_active),
            "worker_limit": worker_limit,
            "worker_limit_source": worker_limit_source,
            "foreign_compute_allowed": bool(foreign_compute_allowed),
            "max_tasks_per_gpu": int(max_tasks_per_gpu),
            "groups_with_memory_claim": sorted(all_active_keys & claimed_group_keys),
            "groups_waiting_for_memory_claim": sorted(
                all_active_keys - claimed_group_keys
            ),
            "formal_execution_authorization_ready": authorization_ready,
            "terminal_failures": sorted(terminal_failures),
            "waiting_reasons": dict(waiting_reasons),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _write_json(state_path, status)
        if status["status"] in {"ok", "failed"}:
            _write_json(
                _heartbeat_path(final_root),
                _heartbeat_payload(plan["plan_sha256"], poll_seconds, status["status"]),
            )
            return status
        time.sleep(int(poll_seconds))
