#!/usr/bin/env python3
"""Operate the single frozen CropGenome-FM final downstream run."""

import argparse
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.final_assets import (
    prepare_final_datasets, prepare_final_task, refresh_dataset_index,
)
from downstream_v4.environment import write_environment_receipt
from downstream_v4.dataset_audit import audit_all_datasets
from downstream_v4.final_audit import audit_final_closure
from downstream_v4.cpu_scheduler import submit_ready_sensitivity, submit_ready_wave
from downstream_v4.final_controller import (
    CHECKPOINT_ROOT, _checkpoint_record, build_gpu_groups, execute_cpu_row,
    current_controller_status, execute_group, record_cpu_row_failure, run_daemon,
    run_zero_child,
)
from downstream_v4.final_protocol import write_execution_plan
from downstream_v4.formal_gate import (
    authorize_formal_execution, valid_formal_execution_authorization,
)
from downstream_v4.final_report import build_final_report
from downstream_v4.low_homology import build_task_cohort
from downstream_v4.model_smoke import run_public_model_smoke, run_smoke_daemon

from downstream_v4.sensitivity import execute_ready_sensitivity_rows, execute_sensitivity_row
from downstream_v4.registry import (
    FORBIDDEN_INTERNAL_MODEL_IDS, INTERNAL_PRETRAINING_ABLATIONS,
    LICENSE_EXECUTION_POLICY, load_registry,
)


DEFAULT_FINAL_ROOT = "training_server_transfer/runs/cropgenome_downstream_final_v1"
DEFAULT_REGISTRY = "training_server_transfer/configs/cropgenome_downstream_v4.json"


def _assert_active_scientific_policy(registry, plan):
    registry_policy = registry.get("policy") or {}
    comparison = plan.get("comparison_policy") or {}
    forbidden = set(plan.get("model_ids") or []) & FORBIDDEN_INTERNAL_MODEL_IDS
    forbidden.update(
        model.get("model_id") for model in registry.get("models") or []
        if model.get("kind") == "internal_control"
        or model.get("model_id") in FORBIDDEN_INTERNAL_MODEL_IDS
    )
    if forbidden:
        raise RuntimeError(
            "internal pretraining ablations are disabled by user: "
            + ", ".join(sorted(value for value in forbidden if value))
        )
    if (
        registry_policy.get("license_execution_policy") != LICENSE_EXECUTION_POLICY
        or registry_policy.get("license_metadata_is_execution_gate") is not False
        or registry_policy.get("internal_pretraining_ablations")
        != INTERNAL_PRETRAINING_ABLATIONS
        or comparison.get("license_execution_policy") != LICENSE_EXECUTION_POLICY
        or comparison.get("license_metadata_is_execution_gate") is not False
        or comparison.get("internal_pretraining_ablations")
        != INTERNAL_PRETRAINING_ABLATIONS
    ):
        raise RuntimeError("active downstream scientific policy regressed")


def _load(project_root, final_root, registry_path):
    project_root = Path(project_root).resolve()
    final_root = Path(final_root)
    if not final_root.is_absolute():
        final_root = project_root / final_root
    registry_path = Path(registry_path)
    if not registry_path.is_absolute():
        registry_path = project_root / registry_path
    registry = load_registry(registry_path)
    plan_path = final_root / "FINAL_PROTOCOL.json"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _assert_active_scientific_policy(registry, plan)
    return project_root, final_root.resolve(), registry, plan


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    prepare = sub.add_parser("prepare-datasets")
    prepare.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    prepare_one = sub.add_parser("prepare-task")
    prepare_one.add_argument("--task-id", required=True)
    prepare_one.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    refresh = sub.add_parser("refresh-dataset-index")
    refresh.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    group = sub.add_parser("run-group")
    group.add_argument("--group-key", required=True)
    group.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    group.add_argument("--allow-foreign-compute", action="store_true")
    group.add_argument("--max-tasks-per-gpu", type=int, default=1)
    zero = sub.add_parser("zero-child")
    zero.add_argument("--spec", required=True)
    daemon = sub.add_parser("daemon")
    daemon.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    daemon.add_argument(
        "--max-workers", type=int, default=0,
        help="0 uses the frozen plan max_workers; positive values may only tighten it",
    )
    daemon.add_argument("--poll-seconds", type=int, default=60)
    daemon.add_argument("--max-attempts", type=int, default=3)
    daemon.add_argument("--allow-foreign-compute", action="store_true")
    daemon.add_argument(
        "--max-tasks-per-gpu", type=int, default=1,
        help="must equal the frozen per-GPU task limit",
    )
    cpu = sub.add_parser("cpu-row")
    cpu.add_argument("--run-key", required=True)
    cpu.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    chunk = sub.add_parser("cpu-chunk")
    chunk.add_argument("--manifest", required=True)
    chunk.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    wave = sub.add_parser("cpu-wave")
    wave.add_argument("--manifest", required=True)
    wave.add_argument("--index", type=int)
    wave.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    submit_cpu = sub.add_parser("submit-cpu")
    submit_cpu.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    submit_cpu.add_argument("--max-rows", type=int, default=256)
    submit_cpu.add_argument("--chunk-size", type=int, default=4)
    submit_sensitivity = sub.add_parser("submit-sensitivity")
    submit_sensitivity.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    submit_sensitivity.add_argument("--max-rows", type=int, default=32)
    cohort = sub.add_parser("build-cohort")
    cohort.add_argument("--task-id", required=True, choices=["B13", "B14", "B15", "B16"])
    cohort.add_argument("--threads", type=int, default=8)
    cohort.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    model_smoke = sub.add_parser("run-model-smoke")
    model_smoke.add_argument("--model-id", required=True)
    model_smoke.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    smoke_daemon = sub.add_parser("model-smoke-daemon")
    smoke_daemon.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    smoke_daemon.add_argument("--max-workers", type=int, default=3)
    smoke_daemon.add_argument("--poll-seconds", type=int, default=60)
    audit = sub.add_parser("audit")
    audit.add_argument("--deep", action="store_true")
    audit.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    report = sub.add_parser("final-report")
    report.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    environment = sub.add_parser("environment-receipt")
    environment.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    dataset_audit = sub.add_parser("audit-datasets")
    dataset_audit.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)

    formal_authorization = sub.add_parser("authorize-formal-execution")
    formal_authorization.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)

    sensitivity = sub.add_parser("run-sensitivity")
    sensitivity.add_argument("--run-key", required=True)
    sensitivity.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    ready_sensitivity = sub.add_parser("run-ready-sensitivity")
    ready_sensitivity.add_argument("--max-rows", type=int, default=32)
    ready_sensitivity.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    checkpoint_inventory = sub.add_parser("inventory-checkpoints")
    checkpoint_inventory.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    status = sub.add_parser("status")
    status.add_argument("--final-root", default=DEFAULT_FINAL_ROOT)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "protocol":
        project_root = Path(args.project_root).resolve()
        registry_path = Path(args.registry)
        if not registry_path.is_absolute():
            registry_path = project_root / registry_path
        final_root = Path(args.final_root)
        if not final_root.is_absolute():
            final_root = project_root / final_root
        frozen = write_execution_plan(
            load_registry(registry_path), final_root / "FINAL_PROTOCOL.json",
        )
        result = {
            "status": "ok" if frozen["checkpoint_identity_state"] == "frozen_complete" else "draft",
            "protocol_id": frozen["protocol_id"], "plan_sha256": frozen["plan_sha256"],
            "rows": len(frozen["rows"]),
            "applicability_entries": len(frozen["applicability"]),
            "checkpoint_identity_state": frozen["checkpoint_identity_state"],
            "checkpoint_identities": len(frozen["checkpoint_identities"]),
            "path": str(final_root / "FINAL_PROTOCOL.json"),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 0
    if args.command == "zero-child":
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        project_root = Path(spec["project_root"])
        registry = load_registry(project_root / DEFAULT_REGISTRY)
        result = run_zero_child(args.spec, registry)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 0
    project_root, final_root, registry, plan = _load(
        args.project_root, args.final_root, args.registry,
    )
    formal_commands = {
        "run-group", "daemon", "cpu-row", "cpu-chunk", "cpu-wave", "submit-cpu",
        "run-sensitivity", "run-ready-sensitivity", "submit-sensitivity",
    }
    if (
        args.command in formal_commands
        and plan.get("checkpoint_identity_state") != "frozen_complete"
    ):
        raise RuntimeError(
            "formal execution requires frozen identities for all scheduled checkpoints"
        )
    if (
        args.command in formal_commands
        and valid_formal_execution_authorization(final_root, plan["plan_sha256"]) is None
    ):
        raise RuntimeError("formal execution authorization missing or stale")
    if args.command == "prepare-datasets":
        result = prepare_final_datasets(registry, project_root, final_root)
    elif args.command == "prepare-task":
        result = prepare_final_task(registry, args.task_id, project_root, final_root)
    elif args.command == "refresh-dataset-index":
        result = refresh_dataset_index(registry, final_root)
    elif args.command == "authorize-formal-execution":
        result = authorize_formal_execution(
            plan, registry, project_root, final_root,
        )
    elif args.command == "run-group":
        groups = {group["group_key"]: group for group in build_gpu_groups(plan)}
        if args.group_key not in groups:
            raise SystemExit(f"unknown group key: {args.group_key}")
        result = execute_group(
            groups[args.group_key], registry, project_root, final_root,
            plan["plan_sha256"], Path(__file__).resolve(),
            foreign_compute_allowed=args.allow_foreign_compute,
            max_tasks_per_gpu=args.max_tasks_per_gpu,
        )
    elif args.command == "daemon":
        result = run_daemon(
            plan, registry, project_root, final_root, Path(__file__).resolve(),
            args.max_workers, args.poll_seconds, args.max_attempts,
            foreign_compute_allowed=args.allow_foreign_compute,
            max_tasks_per_gpu=args.max_tasks_per_gpu,
        )
    elif args.command in {"cpu-row", "cpu-chunk", "cpu-wave"}:
        rows = {row["run_key"]: row for row in plan["rows"]}
        if args.command == "cpu-row":
            run_keys = [args.run_key]
        elif args.command == "cpu-chunk":
            run_keys = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["run_keys"]
        else:
            import os
            payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
            run_keys = payload["chunks"][index]
        completed = []; failures = []
        for run_key in run_keys:
            if run_key not in rows:
                raise KeyError(f"unknown run key: {run_key}")
            try:
                completed.append(execute_cpu_row(
                    rows[run_key], project_root, final_root, plan["plan_sha256"],
                    seeds=plan["seeds"],
                ))
            except Exception:
                failure = record_cpu_row_failure(
                    rows[run_key], final_root, plan["plan_sha256"], traceback.format_exc(),
                )
                failures.append({"run_key": run_key, "status": failure["status"]})
        result = {
            "status": "failed" if failures else "ok",
            "completed": len(completed), "failed": failures, "run_keys": run_keys,
        }
    elif args.command == "submit-cpu":
        result = submit_ready_wave(
            project_root, final_root, plan,
            max_rows=args.max_rows, chunk_size=args.chunk_size,
        )
    elif args.command == "submit-sensitivity":
        result = submit_ready_sensitivity(
            project_root, final_root, plan, registry, max_rows=args.max_rows,
        )
    elif args.command == "build-cohort":
        result = build_task_cohort(
            args.task_id, final_root / "datasets" / args.task_id,
            final_root / "low_homology", threads=args.threads,
        )
    elif args.command == "run-model-smoke":
        result = run_public_model_smoke(
            registry, args.model_id, project_root, final_root,
        )
    elif args.command == "model-smoke-daemon":
        result = run_smoke_daemon(
            registry, project_root, final_root, Path(__file__).resolve(),
            max_workers=args.max_workers, poll_seconds=args.poll_seconds,
        )
    elif args.command == "audit":
        result = audit_final_closure(
            plan, registry, project_root, final_root, deep=args.deep,
        )
    elif args.command == "final-report":
        result = build_final_report(plan, registry, project_root, final_root)
    elif args.command == "environment-receipt":
        result = write_environment_receipt(
            project_root / "training_server_transfer/configs/downstream_v4_environment.lock.json",
            final_root / "ENVIRONMENT_RECEIPT.json", plan["plan_sha256"],
        )
    elif args.command == "audit-datasets":
        result = audit_all_datasets(registry, final_root)

    elif args.command == "run-sensitivity":
        row = next(row for row in plan["rows"] if row["run_key"] == args.run_key)
        result = execute_sensitivity_row(row, registry, final_root, plan["plan_sha256"])
    elif args.command == "run-ready-sensitivity":
        result = execute_ready_sensitivity_rows(
            plan["rows"], registry, final_root, plan["plan_sha256"],
            max_rows=args.max_rows,
        )
    elif args.command == "inventory-checkpoints":
        records = []; missing = []
        for step in sorted({
            int(row["checkpoint_step"]) for row in plan["rows"]
            if row.get("checkpoint_scope") == "step" and row.get("checkpoint_step") is not None
        }):
            checkpoint = ROOT / CHECKPOINT_ROOT / f"step_{step:08d}.pt"
            if checkpoint.is_file():
                records.append(_checkpoint_record(checkpoint, final_root))
            else:
                missing.append(step)
        result = {"status": "ok", "verified": len(records), "missing_steps": missing, "records": records}
    elif args.command == "status":
        result = current_controller_status(plan, final_root)
    else:
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0 if result.get("status") not in {"failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
