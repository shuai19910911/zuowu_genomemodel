"""Frozen final downstream protocol and deterministic execution matrix."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .feature_projection import HEAD_INPUT_DIM, PROJECTION_PROTOCOL_ID
from .low_homology import LOW_HOMOLOGY_PROTOCOL_ID
from .model_adapters import MODEL_RUNTIME_SPECS
from .model_smoke import MODEL_SMOKE_PROTOCOL_ID
from .probe import PROBE_PROTOCOL_ID
from .registry import (
    FORBIDDEN_INTERNAL_MODEL_IDS, INTERNAL_PRETRAINING_ABLATIONS,
    LICENSE_EXECUTION_POLICY,
)
from .sensitivity import SENSITIVITY_PROTOCOL_ID
from .token_probe import TOKEN_PROBE_PROTOCOL_ID
from .zero_shot import ZERO_SHOT_PROTOCOL_ID


PROTOCOL_ID = "cropgenome_downstream_final_v1"
EXCLUDED_TASK_IDS = ("B11", "B12")
SEEDS = (13, 29, 43, 71, 97)
EVIDENCE_SCOPE_BY_TASK = {
    "A04": "official_within_species_split_not_cross_species_generalization",
}

IMPLEMENTATION_ASSETS = (
    "training_server_transfer/downstream_v4/article_analysis.py",
    "training_server_transfer/downstream_v4/article_protocol.py",
    "training_server_transfer/downstream_v4/adapters.py",
    "training_server_transfer/downstream_v4/baselines.py",
    "training_server_transfer/downstream_v4/checkpoint_launcher.py",
    "training_server_transfer/downstream_v4/commands.py",
    "training_server_transfer/downstream_v4/cpu_scheduler.py",
    "training_server_transfer/downstream_v4/data.py",
    "training_server_transfer/downstream_v4/dataset_audit.py",
    "training_server_transfer/downstream_v4/download.py",
    "training_server_transfer/downstream_v4/environment.py",
    "training_server_transfer/downstream_v4/feature_projection.py",
    "training_server_transfer/downstream_v4/final_audit.py",
    "training_server_transfer/downstream_v4/final_assets.py",
    "training_server_transfer/downstream_v4/final_controller.py",
    "training_server_transfer/downstream_v4/formal_gate.py",
    "training_server_transfer/downstream_v4/final_protocol.py",
    "training_server_transfer/downstream_v4/final_report.py",
    "training_server_transfer/downstream_v4/gpu_gate.py",
    "training_server_transfer/downstream_v4/low_homology.py",
    "training_server_transfer/downstream_v4/metrics.py",
    "training_server_transfer/downstream_v4/model_adapters.py",
    "training_server_transfer/downstream_v4/model_smoke.py",
    "training_server_transfer/downstream_v4/operational_amendment.py",
    "training_server_transfer/downstream_v4/preparation.py",
    "training_server_transfer/downstream_v4/probe.py",
    "training_server_transfer/downstream_v4/registry.py",
    "training_server_transfer/downstream_v4/sensitivity.py",
    "training_server_transfer/downstream_v4/streaming_embeddings.py",
    "training_server_transfer/downstream_v4/token_probe.py",
    "training_server_transfer/downstream_v4/zero_shot.py",
    "scripts/extract_cropgenome_bench_v1_embeddings.py",
    "scripts/extract_evo2_embeddings.py",
    "scripts/extract_public_dna_embeddings.py",
    "scripts/extract_public_token_embeddings.py",
    "scripts/launch_cropgenome_downstream_checkpoint.py",
    "scripts/run_cropgenome_downstream_final.py",
    "scripts/summarize_cropgenome_article_results.py",
    "training_server_transfer/scripts/train.py",
    "training_server_transfer/configs/downstream_v4_environment.lock.json",
    "training_server_transfer/configs/downstream_article_gpu05_2080ti.json",
    "training_server_transfer/configs/downstream_article_guided_v1.json",
    "training_server_transfer/configs/model_large.json",
)


def checkpoint_steps(start=16000, stop=50000, interval=2000):
    return list(range(int(start), int(stop) + 1, int(interval)))


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_assets():
    project_root = Path(__file__).resolve().parents[2]
    records = {}
    for relative in IMPLEMENTATION_ASSETS:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records[relative] = {
            "sha256": _sha256_path(path), "size_bytes": path.stat().st_size,
        }
    return records


def _model_capacity(model):
    if model["kind"] == "public_weight":
        return int(MODEL_RUNTIME_SPECS[model["model_id"]]["context_bp"])
    if model["kind"] in {"project_model", "simple_baseline"}:
        return 8192
    raise ValueError(f"unsupported final model kind: {model['kind']}")


def _supports_task(model, task):
    model_id = model["model_id"]
    capabilities = set(model.get("capabilities") or [])
    task_id, task_kind = task["task_id"], task["task_kind"]
    if task_id == "B17":
        if model["kind"] == "public_weight":
            return bool(MODEL_RUNTIME_SPECS[model_id]["token_aligned"])
        return "token_embedding" in capabilities
    if task_kind == "token_multiclass":
        if model["kind"] == "public_weight":
            return bool(MODEL_RUNTIME_SPECS[model_id]["token_aligned"])
        return "token_embedding" in capabilities
    if task_kind.startswith("zero_shot"):
        if model_id == "CropGenomeFM":
            return "masked_lm" in capabilities or "variant_scoring" in capabilities
        if model["kind"] == "public_weight":
            return bool(MODEL_RUNTIME_SPECS[model_id]["zero_shot"])
        return False
    return "embedding" in capabilities


def _contexts(model, task):
    declared = [int(value) for value in task.get("context_bp") or [512]]
    if task["task_id"] == "C28":
        declared = [8192]
    capacity = _model_capacity(model)
    return [value for value in declared if value <= capacity]


def _cv_policy(task):
    if str(task.get("task_kind") or "").startswith("zero_shot"):
        return "official_test_only_zero_shot"
    if task.get("split_policy") == "official_leave_one_chromosome_out":
        return "leave_one_group_out"
    return "fixed_train_validation_test"


def _row(model, task, context_bp, checkpoint_step=None):
    scope = "step" if checkpoint_step is not None else "shared"
    scope_key = f"step{int(checkpoint_step):05d}" if checkpoint_step is not None else "shared"
    model_id, task_id = model["model_id"], task["task_id"]
    run_key = f"{scope_key}__{task_id}__{model_id}__ctx{int(context_bp)}"
    evaluation_protocol_id = PROBE_PROTOCOL_ID
    if task_id == "B13":
        evaluation_protocol_id = TOKEN_PROBE_PROTOCOL_ID
    elif task["task_kind"].startswith("zero_shot"):
        evaluation_protocol_id = ZERO_SHOT_PROTOCOL_ID
    elif task_id == "B17":
        evaluation_protocol_id = SENSITIVITY_PROTOCOL_ID
    split_policy = task.get("split_policy")
    cv_policy = _cv_policy(task)
    return {
        "run_key": run_key,
        "checkpoint_scope": scope,
        "checkpoint_step": int(checkpoint_step) if checkpoint_step is not None else None,
        "task_id": task_id,
        "task_kind": task["task_kind"],
        "primary_metric": task["primary_metric"],
        "primary_metric_direction": "higher_is_better",
        "model_id": model_id,
        "model_kind": model["kind"],
        "context_bp": int(context_bp),
        "profile": "full",
        "split_policy": split_policy,
        "cv_policy": cv_policy,
        "evidence_scope": (
            "official_test_only_zero_shot_evidence"
            if task["task_kind"].startswith("zero_shot")
            else EVIDENCE_SCOPE_BY_TASK.get(
                task_id, "registry_declared_split_generalization",
            )
        ),
        "seeds": list(SEEDS),
        "execution_kind": "analysis" if task_id == "B17" else "evaluation",
        "evaluation_protocol_id": evaluation_protocol_id,
        "bootstrap_unit": (
            "candidate_group" if task["task_kind"] == "candidate_ranking"
            else "held_in_group" if cv_policy == "leave_one_group_out"
            else "stratified_row"
        ),
    }


def build_execution_plan(registry, checkpoint_identities=None,
                         checkpoint_identity_state="draft_unbound",
                         excluded_task_ids=EXCLUDED_TASK_IDS):
    excluded_task_ids = tuple(sorted(set(map(str, excluded_task_ids or ()))))
    tasks = [
        task for task in registry["tasks"]
        if task["task_id"] not in excluded_task_ids
    ]
    models = list(registry["models"])
    forbidden_ids = sorted(
        model["model_id"] for model in models
        if model.get("model_id") in FORBIDDEN_INTERNAL_MODEL_IDS
    )
    forbidden_kinds = sorted(
        model["model_id"] for model in models if model.get("kind") == "internal_control"
    )
    if forbidden_ids or forbidden_kinds:
        raise RuntimeError(
            "internal pretraining ablations are disabled by user: "
            + ", ".join(sorted(set(forbidden_ids + forbidden_kinds)))
        )
    registry_policy = registry.get("policy") or {}
    if (
        registry_policy.get("license_execution_policy") != LICENSE_EXECUTION_POLICY
        or registry_policy.get("license_metadata_is_execution_gate") is not False
        or registry_policy.get("internal_pretraining_ablations")
        != INTERNAL_PRETRAINING_ABLATIONS
    ):
        raise RuntimeError("active registry scientific policy regressed")
    rows = []
    applicability = []
    for task in tasks:
        for model in models:
            supported = _supports_task(model, task)
            contexts = _contexts(model, task) if supported else []
            declared_contexts = [int(value) for value in task.get("context_bp") or [512]]
            applicability.append({
                "task_id": task["task_id"], "model_id": model["model_id"],
                "status": "applicable" if contexts else "not_applicable",
                "reason": (
                    None if contexts else
                    "capability_not_supported" if not supported else "context_exceeds_model_capacity"
                ),
                "declared_context_bp": declared_contexts,
                "eligible_context_bp": contexts,
                "excluded_context_bp": sorted(set(declared_contexts) - set(contexts)),
            })
            if not supported:
                continue
            if model["model_id"] == "CropGenomeFM":
                for step in checkpoint_steps():
                    for context_bp in contexts:
                        rows.append(_row(model, task, context_bp, checkpoint_step=step))
            else:
                for context_bp in contexts:
                    rows.append(_row(model, task, context_bp))
    rows.sort(key=lambda row: row["run_key"])
    task_ids = [task["task_id"] for task in tasks]
    registry_canonical = json.dumps(
        registry, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    runtime_canonical = json.dumps(
        MODEL_RUNTIME_SPECS, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    plan = {
        "protocol_id": PROTOCOL_ID,
        "registry_version": registry["registry_version"],
        "registry_semantic_sha256": hashlib.sha256(registry_canonical.encode("utf-8")).hexdigest(),
        "model_runtime_semantic_sha256": hashlib.sha256(runtime_canonical.encode("utf-8")).hexdigest(),
        "implementation_assets": _implementation_assets(),
        "profile": "full",
        "checkpoint_schedule": {
            "start": 16000, "stop": 50000, "interval": 2000,
            "steps": checkpoint_steps(),
        },
        "checkpoint_identity_state": checkpoint_identity_state,
        "checkpoint_identities": checkpoint_identities or {},
        "task_ids": task_ids,
        "excluded_task_ids": list(excluded_task_ids),
        "model_ids": [model["model_id"] for model in models],
        "seeds": list(SEEDS),
        "probe_protocols": {
            "pooled": PROBE_PROTOCOL_ID,
            "token": TOKEN_PROBE_PROTOCOL_ID,
            "zero_shot": ZERO_SHOT_PROTOCOL_ID,
            "sensitivity": SENSITIVITY_PROTOCOL_ID,
            "low_homology": LOW_HOMOLOGY_PROTOCOL_ID,
            "model_smoke": MODEL_SMOKE_PROTOCOL_ID,
            "feature_projection": PROJECTION_PROTOCOL_ID,
        },
        "comparison_policy": {
            "same_dataset": True,
            "same_split": True,
            "same_context_within_track": True,
            "same_head_and_budget": True,
            "head_input_dim": HEAD_INPUT_DIM,
            "shared_public_baselines_reused_by_receipt_hash": True,
            "public_weights_real_rerun": True,
            "license_execution_policy": LICENSE_EXECUTION_POLICY,
            "license_metadata_is_execution_gate": False,
            "internal_pretraining_ablations": INTERNAL_PRETRAINING_ABLATIONS,
            "headline_evidence": "crop_specific_tasks_with_public_model_comparisons",
        },
        "test_policy": {
            "metrics_sealed_until_matrix_complete": True,
            "checkpoint_selected_from_test": False,
            "checkpoint_trajectory_is_monitoring_evidence": True,
            "final_checkpoint_is_predeclared_step_50000": True,
            "multiplicity_correction_required": True,
        },
        "no_intermediate_release": True,
        "applicability": applicability,
        "rows": rows,
    }
    canonical = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    plan["plan_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def _checkpoint_identities(output_path):
    project_root = Path(__file__).resolve().parents[2]
    final_root = Path(output_path).resolve().parent
    checkpoint_root = (
        project_root
        / "training_server_transfer/runs/Stage_B_continuation_3gpu_no_replacement_from_step14000/checkpoints"
    )
    expected = {
        f"step_{step:08d}": checkpoint_root / f"step_{step:08d}.pt"
        for step in checkpoint_steps()
    }
    inventory_path = final_root / "CHECKPOINT_INVENTORY.json"
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        records = inventory.get("records") or {}
    except (OSError, ValueError, AttributeError):
        records = {}
    identities = {}
    for identity, path in expected.items():
        resolved = path.resolve()
        record = records.get(str(resolved))
        if not resolved.is_file() or not record:
            continue
        stat = resolved.stat()
        if (
            int(record.get("size_bytes", -1)) != stat.st_size
            or int(record.get("mtime_ns", -1)) != stat.st_mtime_ns
            or len(str(record.get("sha256", ""))) != 64
        ):
            continue
        identities[identity] = {
            "path": str(resolved), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": record["sha256"],
        }
    state = "frozen_complete" if set(identities) == set(expected) else "draft_incomplete"
    return identities, state


def write_execution_plan(registry, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    identities, identity_state = _checkpoint_identities(output_path)
    plan = build_execution_plan(
        registry, checkpoint_identities=identities,
        checkpoint_identity_state=identity_state,
    )
    plan["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return plan
