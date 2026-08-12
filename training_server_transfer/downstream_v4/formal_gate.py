"""Plan-bound authorization receipt for delayed formal execution workers."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .registry import (
    FORBIDDEN_INTERNAL_MODEL_IDS, INTERNAL_PRETRAINING_ABLATIONS,
    LICENSE_EXECUTION_POLICY,
)
from .operational_amendment import (
    AMENDMENT_FILENAME, valid_memory_optimization_amendment,
)


FORMAL_AUTHORIZATION_PROTOCOL_ID = "cropgenome_formal_execution_authorization_v1"
FORMAL_AUTHORIZATION_FILENAME = "FORMAL_EXECUTION_AUTHORIZATION.json"


def _expected_dataset_task_ids(plan, registry):
    from .article_protocol import ARTICLE_GUIDED_PROTOCOL_ID
    from .final_protocol import EXCLUDED_TASK_IDS

    selected = set(map(str, plan.get("task_ids") or ()))
    available = {str(task["task_id"]) for task in registry.get("tasks") or []}
    if not selected:
        selected = available
    if not selected.issubset(available):
        raise RuntimeError("plan references registry tasks that do not exist")
    excluded = (
        {"B17"}
        if plan.get("protocol_id") == ARTICLE_GUIDED_PROTOCOL_ID
        else set(EXCLUDED_TASK_IDS) | {"B17"}
    )
    return selected - excluded


def _rebuild_live_plan(
    plan, registry, execution_plan_builder=None, checkpoint_set_builder=None,
):
    from .article_protocol import (
        ARTICLE_GUIDED_PROTOCOL_ID, build_article_guided_checkpoint_plan,
    )
    if execution_plan_builder is None:
        from .final_protocol import build_execution_plan
        execution_plan_builder = build_execution_plan
    if checkpoint_set_builder is None:
        from .latest_snapshot import build_checkpoint_set_plan
        checkpoint_set_builder = build_checkpoint_set_plan
    from .latest_snapshot import CHECKPOINT_SET_PROTOCOL_ID

    if plan.get("protocol_id") == ARTICLE_GUIDED_PROTOCOL_ID:
        return build_article_guided_checkpoint_plan(
            registry,
            checkpoint_identities=plan.get("checkpoint_identities") or {},
            project_model_config=plan.get("project_model_config") or {},
            gpu_scheduler_policy=plan.get("gpu_scheduler_policy") or {},
            profile=plan.get("article_guided_evaluation") or {},
        )
    if plan.get("protocol_id") == CHECKPOINT_SET_PROTOCOL_ID:
        return checkpoint_set_builder(
            registry,
            plan.get("checkpoint_identities") or {},
            plan.get("project_model_config") or {},
            plan.get("gpu_scheduler_policy") or {},
        )
    return execution_plan_builder(
        registry,
        checkpoint_identities=plan.get("checkpoint_identities") or {},
        checkpoint_identity_state=plan.get("checkpoint_identity_state", "draft_unbound"),
    )


def _implementation_drift_evidence(plan, live_plan, project_root, final_root):
    if live_plan.get("plan_sha256") == plan.get("plan_sha256"):
        return []
    amendment = valid_memory_optimization_amendment(
        final_root, plan, live_plan, project_root,
    )
    if amendment is None:
        return None
    paths = [Path(final_root).resolve() / AMENDMENT_FILENAME]
    paths.extend(
        Path(record["path"]).resolve()
        for record in amendment.get("supporting_evidence") or []
    )
    return paths


def _valid_migrated_model_smoke(
    amendment, final_root, model_id, current_contract_sha256, receipt_validator,
):
    migration_record = next((
        record for record in (amendment or {}).get("supporting_evidence") or []
        if Path(record.get("path", "")).name
        == "MEMORY_OPTIMIZATION_SMOKE_MIGRATION.json"
    ), None)
    if migration_record is None:
        return False
    migration_path = Path(migration_record.get("path", ""))
    try:
        migration = json.loads(migration_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    records = migration.get("records") or []
    matches = [record for record in records if record.get("model_id") == model_id]
    if not (
        migration.get("status") == "ok"
        and migration.get("protocol_id")
        == "memory_optimization_smoke_contract_migration_v1"
        and int(migration.get("models_total", 0)) == 14
        and len(records) == 14
        and len(matches) == 1
        and matches[0].get("current_contract_sha256") == current_contract_sha256
    ):
        return False
    row = matches[0]
    receipt_path = (
        Path(final_root).resolve() / "model_smokes" / model_id / "SMOKE_RECEIPT.json"
    )
    try:
        stat = receipt_path.stat()
    except OSError:
        return False
    if not (
        Path(row.get("legacy_receipt_path", "")).resolve() == receipt_path
        and stat.st_size == int(row.get("legacy_receipt_size_bytes", -1))
        and _sha256_path(receipt_path) == row.get("legacy_receipt_sha256")
    ):
        return False
    legacy_contract = row.get("legacy_contract_sha256")
    return receipt_validator(final_root, model_id, legacy_contract) is not None


def _safe_gpu_scheduler_policy(plan):
    """Strict dynamic mode is safe; packed mode must still use one idle UUID."""
    policy = plan.get("gpu_scheduler_policy") or {}
    if not policy:
        return True
    if policy.get("mode") != "memory_packed":
        return False
    try:
        max_tasks = int(policy.get("max_tasks_per_gpu", 0))
    except (TypeError, ValueError):
        return False
    return (
        max_tasks == 1
        and policy.get("foreign_compute_allowed") is False
        and policy.get("unknown_compute_blocks") is True
        and policy.get("nvitop_is_benign") is True
        and policy.get("required_hostname") == "gpu05"
        and list(policy.get("allowed_gpu_indices") or []) == list(range(7))
    )


def _canonical_sha(payload):
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _sha256_path(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_formal_execution_authorization_receipt(final_root, plan, evidence_paths, gate_counts):
    if plan.get("checkpoint_identity_state") != "frozen_complete":
        raise RuntimeError("formal authorization requires checkpoint_identity_state=frozen_complete")
    comparison = plan.get("comparison_policy") or {}
    if (
        comparison.get("license_execution_policy") != LICENSE_EXECUTION_POLICY
        or comparison.get("license_metadata_is_execution_gate") is not False
        or comparison.get("internal_pretraining_ablations")
        != INTERNAL_PRETRAINING_ABLATIONS
        or set(plan.get("model_ids") or []) & FORBIDDEN_INTERNAL_MODEL_IDS
        or not _safe_gpu_scheduler_policy(plan)
    ):
        raise RuntimeError("formal authorization refuses regressed scientific policy")
    plan_sha256 = str(plan.get("plan_sha256") or "")
    if not plan_sha256:
        raise RuntimeError("formal authorization requires a plan_sha256")
    evidence = []
    for raw_path in sorted({str(Path(path).resolve()) for path in evidence_paths}):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        evidence.append({
            "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": _sha256_path(path),
        })
    checkpoints = []
    for identity_key, identity in sorted((plan.get("checkpoint_identities") or {}).items()):
        path = Path(identity.get("path", "")).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        if (
            stat.st_size != int(identity.get("size_bytes", -1))
            or stat.st_mtime_ns != int(identity.get("mtime_ns", -1))
            or not identity.get("sha256")
        ):
            raise RuntimeError(f"checkpoint identity drift: {identity_key}")
        checkpoints.append({
            "identity_key": identity_key, "path": str(path),
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": identity["sha256"],
        })
    payload = {
        "status": "authorized",
        "protocol_id": FORMAL_AUTHORIZATION_PROTOCOL_ID,
        "plan_sha256": plan_sha256,
        "checkpoint_identity_state": "frozen_complete",
        "evidence": evidence,
        "checkpoints": checkpoints,
        "gate_counts": dict(sorted((gate_counts or {}).items())),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    payload["receipt_sha256"] = _canonical_sha(payload)
    path = Path(final_root).resolve() / FORMAL_AUTHORIZATION_FILENAME
    _atomic_json(path, payload)
    return payload


def valid_formal_execution_authorization(final_root, expected_plan_sha256):
    path = Path(final_root).resolve() / FORMAL_AUTHORIZATION_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    canonical = dict(payload)
    stored = canonical.pop("receipt_sha256", None)
    if not (
        payload.get("status") == "authorized"
        and payload.get("protocol_id") == FORMAL_AUTHORIZATION_PROTOCOL_ID
        and payload.get("plan_sha256") == expected_plan_sha256
        and payload.get("checkpoint_identity_state") == "frozen_complete"
        and stored == _canonical_sha(canonical)
    ):
        return None
    for record in payload.get("evidence") or []:
        path = Path(record.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size_bytes", -1))
            or _sha256_path(path) != record.get("sha256")
        ):
            return None
    for record in payload.get("checkpoints") or []:
        path = Path(record.get("path", ""))
        if not path.is_file():
            return None
        stat = path.stat()
        if (
            stat.st_size != int(record.get("size_bytes", -1))
            or stat.st_mtime_ns != int(record.get("mtime_ns", -1))
            or not record.get("sha256")
        ):
            return None
    return payload


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_formal_prerequisites(plan, registry, project_root, final_root):
    from .dataset_audit import valid_dataset_audit
    from .environment import valid_environment_receipt
    from .model_smoke import smoke_contract_sha256, valid_smoke_receipt


    project_root = Path(project_root).resolve()
    final_root = Path(final_root).resolve()
    blockers = []
    evidence = []

    def require_json(relative, expected_status=None):
        path = final_root / relative
        try:
            payload = _read_json(path)
        except (OSError, ValueError):
            blockers.append(f"missing_or_invalid:{relative}")
            return {}, path
        if expected_status is not None and payload.get("status") != expected_status:
            blockers.append(f"status:{relative}:{payload.get('status')}")
        evidence.append(path)
        return payload, path

    if plan.get("checkpoint_identity_state") != "frozen_complete":
        blockers.append("checkpoint_identity_state")
    comparison = plan.get("comparison_policy") or {}
    if comparison.get("license_execution_policy") != LICENSE_EXECUTION_POLICY:
        blockers.append("license_execution_policy")
    if comparison.get("license_metadata_is_execution_gate") is not False:
        blockers.append("license_metadata_execution_gate")
    if (
        comparison.get("internal_pretraining_ablations")
        != INTERNAL_PRETRAINING_ABLATIONS
    ):
        blockers.append("internal_pretraining_ablation_policy")
    if set(plan.get("model_ids") or []) & FORBIDDEN_INTERNAL_MODEL_IDS:
        blockers.append("forbidden_internal_models_in_plan")
    if not _safe_gpu_scheduler_policy(plan):
        blockers.append("unsafe_gpu_scheduler_policy")
    if any(
        model.get("kind") == "internal_control"
        or model.get("model_id") in FORBIDDEN_INTERNAL_MODEL_IDS
        for model in registry.get("models") or []
    ):
        blockers.append("forbidden_internal_models_in_registry")
    registry_policy = registry.get("policy") or {}
    if (
        registry_policy.get("license_execution_policy") != LICENSE_EXECUTION_POLICY
        or registry_policy.get("license_metadata_is_execution_gate") is not False
        or registry_policy.get("internal_pretraining_ablations")
        != INTERNAL_PRETRAINING_ABLATIONS
    ):
        blockers.append("registry_scientific_policy")
    live = _rebuild_live_plan(plan, registry)
    implementation_evidence = _implementation_drift_evidence(
        plan, live, project_root, final_root,
    )
    memory_amendment = valid_memory_optimization_amendment(
        final_root, plan, live, project_root,
    ) if live.get("plan_sha256") != plan.get("plan_sha256") else None
    if implementation_evidence is None:
        blockers.append("plan_implementation_drift")
    else:
        evidence.extend(implementation_evidence)
    protocol_payload, _ = require_json("FINAL_PROTOCOL.json")
    if protocol_payload.get("plan_sha256") != plan.get("plan_sha256"):
        blockers.append("protocol_file_drift")

    no_ablation, _ = require_json("controller/NO_ABLATION_DECISION.json", "active")
    if (
        no_ablation.get("scope") != "all_internal_pretraining_and_architecture_ablations"
        or not FORBIDDEN_INTERNAL_MODEL_IDS.issubset(
            set(no_ablation.get("removed_models") or [])
        )
        or no_ablation.get("no_region_launch_authorization_revoked") is not True
    ):
        blockers.append("no_ablation_decision")
    license_policy, _ = require_json("controller/LICENSE_EXECUTION_POLICY.json", "active")
    if (
        license_policy.get("execution_policy") != LICENSE_EXECUTION_POLICY
        or license_policy.get("license_metadata_is_execution_gate") is not False
        or int(license_policy.get("review_required_sources", -1)) != 0
        or int(license_policy.get("review_required_models", -1)) != 0
        or int(license_policy.get("formal_tasks_blocked_by_license", -1)) != 0
    ):
        blockers.append("license_execution_policy_decision")
    if (final_root / "LICENSE_OVERRIDE_RECEIPT.json").exists():
        blockers.append("retired_license_override_receipt_active")
    if (final_root / "no_region").exists():
        blockers.append("retired_no_region_namespace_active")

    dataset_index, _ = require_json("DATASET_INDEX.json", "ready")
    expected_dataset_tasks = _expected_dataset_task_ids(plan, registry)
    expected_datasets = len(expected_dataset_tasks)
    ready_dataset_tasks = {
        str(row.get("task_id"))
        for row in dataset_index.get("rows") or []
        if row.get("status") == "ready"
    }
    if not expected_dataset_tasks.issubset(ready_dataset_tasks):
        blockers.append("dataset_index_count")
    dataset_audit, _ = require_json("DATASET_INTEGRITY_AUDIT.json", "ok")
    if valid_dataset_audit(
        registry, final_root, deep=True,
        expected_task_ids=expected_dataset_tasks,
    ) is None:
        blockers.append("dataset_integrity_audit")
    for path in sorted((final_root / "dataset_audits").glob("*.json")):
        evidence.append(path)

    environment_path = final_root / "ENVIRONMENT_RECEIPT.json"
    if valid_environment_receipt(
        environment_path,
        project_root / "training_server_transfer/configs/downstream_v4_environment.lock.json",
        plan["plan_sha256"],
    ) is None:
        blockers.append("environment_receipt")
    elif environment_path.is_file():
        evidence.append(environment_path)

    public_models = [model for model in registry["models"] if model["kind"] == "public_weight"]
    smoke_status, _ = require_json("MODEL_SMOKE_STATUS.json", "ok")
    smoke_ok = 0
    for model in public_models:
        model_id = model["model_id"]
        contract = smoke_contract_sha256(model, project_root)
        current_smoke = valid_smoke_receipt(final_root, model_id, contract)
        migrated_smoke = _valid_migrated_model_smoke(
            memory_amendment, final_root, model_id, contract, valid_smoke_receipt,
        )
        if current_smoke is None and not migrated_smoke:
            blockers.append(f"model_smoke:{model_id}")
            continue
        smoke_ok += 1
        evidence.append(final_root / "model_smokes" / model_id / "SMOKE_RECEIPT.json")
    if int(smoke_status.get("models_complete", -1)) != len(public_models):
        blockers.append("model_smoke_count")

    inventory, _ = require_json("CHECKPOINT_INVENTORY.json")
    expected_checkpoint_count = len((plan.get("checkpoint_schedule") or {}).get("steps") or [])
    if len(plan.get("checkpoint_identities") or {}) != expected_checkpoint_count:
        blockers.append("checkpoint_identity_count")
    if inventory.get("version") != 1 or inventory.get("immutable_checkpoints") is not True:
        blockers.append("checkpoint_inventory_contract")
    if not inventory.get("records"):
        blockers.append("checkpoint_inventory")

    for relative in ("ASSET_QUEUE_SOURCES_STATUS.json", "ASSET_QUEUE_MODELS_STATUS.json"):
        require_json(relative, "ok")

    if blockers:
        raise RuntimeError("formal prerequisites failed: " + ", ".join(sorted(set(blockers))))
    counts = {
        "dataset_records": expected_datasets,
        "model_smokes": smoke_ok,
        "checkpoint_identities": len(plan.get("checkpoint_identities") or {}),
        "dataset_audit_records": len(dataset_audit.get("records") or []),
    }
    return sorted(set(Path(path).resolve() for path in evidence)), counts


def authorize_formal_execution(plan, registry, project_root, final_root):
    evidence, counts = validate_formal_prerequisites(
        plan, registry, project_root, final_root,
    )
    return write_formal_execution_authorization_receipt(
        final_root, plan, evidence, counts,
    )
