#!/usr/bin/env python3
"""Prepare or activate article-guided downstream evaluation for one checkpoint."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.checkpoint_launcher import build_launch_candidate
from downstream_v4.environment import write_environment_receipt
from downstream_v4.formal_gate import authorize_formal_execution


CONFIRMATION = "ACTIVATE_ARTICLE_GUIDED_CHECKPOINT"


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _authorization_status(final_root):
    path = Path(final_root) / "FORMAL_EXECUTION_AUTHORIZATION.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None


def _heartbeat_is_recent(final_root, maximum_age_seconds=180):
    path = Path(final_root) / "controller/GPU_DAEMON_HEARTBEAT.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated = datetime.fromisoformat(str(payload["updated_at"]).replace("Z", "+00:00"))
    except (OSError, ValueError, KeyError):
        return False
    age = (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds()
    return age <= float(maximum_age_seconds)


def _wait_for_matching_heartbeat(
    final_root, plan_sha256, timeout_seconds=120, not_before=None,
):
    path = Path(final_root) / "controller/GPU_DAEMON_HEARTBEAT.json"
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        try:
            updated_at = datetime.fromisoformat(
                str(payload["updated_at"]).replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        except (KeyError, TypeError, ValueError):
            updated_at = None
        fresh_for_launch = (
            updated_at is not None
            and (
                not_before is None
                or updated_at >= not_before.astimezone(timezone.utc)
            )
        )
        if (
            payload.get("status") == "running"
            and payload.get("plan_sha256") == str(plan_sha256)
            and payload.get("hostname") == "gpu05"
            and int(payload.get("pid", 0)) > 0
            and fresh_for_launch
        ):
            return payload
        time.sleep(2)
    raise RuntimeError("gpu05 daemon did not publish a matching live heartbeat")


def _resolve_gpu_host(final_root):
    configured = os.environ.get("CROPGENOME_GPU_HOST", "").strip()
    if configured:
        host = configured
    else:
        path = Path(final_root) / "controller/GPU_HOST.local.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if payload.get("status") == "active" and str(payload.get("host", "")).strip():
            host = str(payload["host"]).strip()
        else:
            host = "gpu05"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", host):
        raise RuntimeError("invalid GPU SSH host")
    return host


def _next_command(args, scheduler_template):
    return shlex.join([
        str(Path(__file__).resolve()),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--final-root", str(Path(args.final_root).resolve()),
        "--registry", str(Path(args.registry).resolve()),
        "--profile", str(Path(args.profile).resolve()),
        "--model-config", str(Path(args.model_config).resolve()),
        "--scheduler-template", str(Path(scheduler_template).resolve()),
        "--execute", "--confirmation", CONFIRMATION,
    ])


def _restore_archived_or_remove(archive, paths):
    archive = Path(archive)
    for path in map(Path, paths):
        backup = archive / path.name
        if backup.is_file():
            shutil.copy2(backup, path)
        elif path.exists():
            path.unlink()


def _activate(candidate, registry_path, confirmation):
    if confirmation != CONFIRMATION:
        raise RuntimeError(f"activation requires --confirmation {CONFIRMATION}")
    final_root = Path(candidate["final_root"]).resolve()
    if _heartbeat_is_recent(final_root):
        raise RuntimeError("a downstream GPU daemon heartbeat is still recent")
    if _authorization_status(final_root) == "authorized":
        raise RuntimeError("current final root is already authorized; pause it before replacing the plan")
    plan_path = final_root / "FINAL_PROTOCOL.json"
    authorization_path = final_root / "FORMAL_EXECUTION_AUTHORIZATION.json"
    environment_path = final_root / "ENVIRONMENT_RECEIPT.json"
    archive = final_root / "controller/plan_archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive.mkdir(parents=True, exist_ok=False)
    if plan_path.is_file():
        shutil.copy2(plan_path, archive / plan_path.name)
    if authorization_path.is_file():
        shutil.copy2(authorization_path, archive / authorization_path.name)
    if environment_path.is_file():
        shutil.copy2(environment_path, archive / environment_path.name)
    try:
        _atomic_json(plan_path, candidate["plan"])
        environment = write_environment_receipt(
            Path(candidate["project_root"])
            / "training_server_transfer/configs/downstream_v4_environment.lock.json",
            final_root / "ENVIRONMENT_RECEIPT.json",
            candidate["plan"]["plan_sha256"],
        )
        if environment.get("status") != "ok":
            raise RuntimeError("runtime environment validation failed")
        registry = json.loads(Path(registry_path).resolve().read_text(encoding="utf-8"))
        authorization = authorize_formal_execution(
            candidate["plan"], registry, Path(candidate["project_root"]), final_root,
        )
        if authorization.get("status") != "authorized":
            raise RuntimeError("formal authorization did not become authorized")
        ssh_command = [
            "ssh", "-f", "-n", "-T",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=120",
            _resolve_gpu_host(final_root),
            candidate["remote_command"],
        ]
        launch_started = datetime.now(timezone.utc)
        subprocess.run(ssh_command, check=True, timeout=180)
        heartbeat = _wait_for_matching_heartbeat(
            final_root, candidate["plan"]["plan_sha256"], timeout_seconds=120,
            not_before=launch_started,
        )
    except Exception:
        _restore_archived_or_remove(
            archive, (plan_path, authorization_path, environment_path),
        )
        raise
    receipt = {
        "protocol_id": candidate["protocol_id"],
        "status": "activated",
        "checkpoint_step": candidate["checkpoint_step"],
        "checkpoint_identity": candidate["checkpoint_identity"],
        "plan_sha256": candidate["plan"]["plan_sha256"],
        "remote_command": candidate["remote_command"],
        "daemon_heartbeat": heartbeat,
        "archive": str(archive),
        "activated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _atomic_json(final_root / "controller/ARTICLE_CHECKPOINT_LAUNCH_RECEIPT.json", receipt)
    return receipt


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare or activate one article-guided CropGenome-FM checkpoint downstream run."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--final-root",
        default=str(ROOT / "training_server_transfer/runs/cropgenome_downstream_final_v1"),
    )
    parser.add_argument(
        "--registry",
        default=str(ROOT / "training_server_transfer/configs/cropgenome_downstream_v4.json"),
    )
    parser.add_argument(
        "--profile",
        default=str(ROOT / "training_server_transfer/configs/downstream_article_guided_v1.json"),
    )
    parser.add_argument(
        "--model-config",
        default=str(ROOT / "training_server_transfer/configs/model_large.json"),
    )
    parser.add_argument("--scheduler-template", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser


def main():
    args = build_parser().parse_args()
    final_root = Path(args.final_root).resolve()
    template = (
        Path(args.scheduler_template).resolve()
        if args.scheduler_template
        else ROOT / "training_server_transfer/configs/downstream_article_gpu05_2080ti.json"
    )
    candidate = build_launch_candidate(
        project_root=ROOT,
        final_root=final_root,
        checkpoint_path=args.checkpoint,
        registry_path=args.registry,
        profile_path=args.profile,
        model_config_path=args.model_config,
        scheduler_template_path=template,
    )
    candidate_path = (
        final_root / "controller/checkpoint_plans"
        / f"article_step_{int(candidate['checkpoint_step']):08d}.json"
    )
    _atomic_json(candidate_path, candidate)
    if args.execute:
        result = _activate(candidate, args.registry, args.confirmation)
    else:
        result = {
            "status": "prepared_not_launched",
            "candidate": str(candidate_path),
            "checkpoint_step": candidate["checkpoint_step"],
            "plan_sha256": candidate["plan"]["plan_sha256"],
            "tasks": len(candidate["plan"]["task_ids"]),
            "next_command": _next_command(args, template),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
