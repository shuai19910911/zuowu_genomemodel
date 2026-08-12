"""Build and activate a one-checkpoint article-guided downstream launch."""

import hashlib
import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path

from .article_protocol import build_article_guided_checkpoint_plan, load_article_profile


CHECKPOINT_LAUNCH_PROTOCOL_ID = "cropgenome_article_checkpoint_launch_v1"
_STEP_PATTERN = re.compile(r"step[_-]?(\d+)", re.IGNORECASE)


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_step(path):
    match = _STEP_PATTERN.search(Path(path).name)
    if match is None:
        raise RuntimeError("checkpoint filename must contain step_<integer>")
    step = int(match.group(1))
    if step <= 0:
        raise RuntimeError("checkpoint step must be positive")
    return step


def _inventory_sha(path, inventory_path):
    path = Path(path).resolve()
    inventory_path = Path(inventory_path).resolve()
    if inventory_path.is_file():
        try:
            records = (json.loads(inventory_path.read_text(encoding="utf-8")).get("records") or {})
            record = records.get(str(path))
            stat = path.stat()
            if (
                record
                and int(record.get("size_bytes", -1)) == stat.st_size
                and int(record.get("mtime_ns", -1)) == stat.st_mtime_ns
                and len(str(record.get("sha256", ""))) == 64
            ):
                return str(record["sha256"]), "checkpoint_inventory"
        except (OSError, ValueError, AttributeError):
            pass
    return _sha256_path(path), "computed_sha256"


def _file_identity(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256_path(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_launch_candidate(
    project_root, final_root, checkpoint_path, registry_path, profile_path,
    model_config_path, scheduler_template_path,
):
    project_root = Path(project_root).resolve()
    final_root = Path(final_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    step = _checkpoint_step(checkpoint_path)
    registry = json.loads(Path(registry_path).resolve().read_text(encoding="utf-8"))
    profile = load_article_profile(profile_path)
    template = json.loads(
        Path(scheduler_template_path).resolve().read_text(encoding="utf-8")
    )
    policy = template.get("policy") or template.get("gpu_scheduler_policy") or {}
    if policy.get("mode") != "memory_packed":
        raise RuntimeError("scheduler template lacks memory_packed policy")
    if (
        int(policy.get("max_workers", 0)) != 3
        or int(policy.get("max_tasks_per_gpu", 0)) != 1
        or policy.get("foreign_compute_allowed") is not False
        or policy.get("unknown_compute_blocks") is not True
    ):
        raise RuntimeError("scheduler template violates frozen gpu05 safety policy")
    checkpoint_sha, identity_source = _inventory_sha(
        checkpoint_path, final_root / "CHECKPOINT_INVENTORY.json",
    )
    stat = checkpoint_path.stat()
    identity_key = f"step_{step:08d}"
    checkpoint_identity = {
        "path": str(checkpoint_path),
        "sha256": checkpoint_sha,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "checkpoint_step": int(step),
        "identity_source": identity_source,
        "protocol_id": "cropgenome_checkpoint_identity_v1",
    }
    model_config = _file_identity(model_config_path)
    plan = build_article_guided_checkpoint_plan(
        registry,
        checkpoint_identities={identity_key: checkpoint_identity},
        project_model_config=model_config,
        gpu_scheduler_policy=policy,
        profile=profile,
    )
    max_workers = int(profile["resource_policy"]["default_max_workers"])
    max_tasks_per_gpu = 1
    python = project_root.parent.parent / ".local/share/mamba/envs/zuowu_genomemodel/bin/python"
    if not python.is_file():
        python = Path("/home/user/zhangzhishuai/.local/share/mamba/envs/zuowu_genomemodel/bin/python")
    runner = project_root / "scripts/run_cropgenome_downstream_final.py"
    command = [
        "PYTHONUNBUFFERED=1", str(python), str(runner),
        "--project-root", str(project_root), "daemon",
        "--final-root", str(final_root),
        "--max-workers", str(max_workers),
        "--max-tasks-per-gpu", str(max_tasks_per_gpu),
        "--poll-seconds", "10",
    ]
    remote_command = " ".join(
        [command[0]] + [shlex.quote(value) for value in command[1:]]
    )
    return {
        "protocol_id": CHECKPOINT_LAUNCH_PROTOCOL_ID,
        "status": "candidate_not_activated",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "final_root": str(final_root),
        "checkpoint_step": int(step),
        "checkpoint_identity": checkpoint_identity,
        "plan": plan,
        "launch_contract": {
            "required_host": "gpu05",
            "max_workers": max_workers,
            "max_tasks_per_gpu": max_tasks_per_gpu,
            "foreign_compute_allowed": False,
            "host_reserved_memory_mib": int(
                profile["resource_policy"]["host_memory_reserved_mib"]
            ),
        },
        "remote_command": remote_command,
    }
