"""Frozen registry loading and semantic validation."""

import json
from pathlib import Path


ALLOWED_ADAPTERS = {
    "canonical_existing", "hf_fasta", "hf_csv", "hf_parquet",
    "hf_tsv", "hf_tsv_token", "hf_parquet_zero_shot", "gpn_parquet",
    "edta_gff", "sensitivity_track",
}
ALLOWED_EXECUTION_STATES = {"implemented", "blocked_upstream_edta"}
FORBIDDEN_SCOPE_TERMS = ("homo sapiens", "mus musculus", "mammal benchmark", "human dataset", "mouse dataset")
FORBIDDEN_INTERNAL_MODEL_IDS = frozenset({
    "CropGenomeFM_no_region", "CropGenomeFM_random_init",
})
LICENSE_EXECUTION_POLICY = "user_authorized_direct_use"
INTERNAL_PRETRAINING_ABLATIONS = "disabled_by_user"


def load_registry(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_registry_path"] = str(path.resolve())
    return payload


def validate_registry(registry, raise_on_error=False):
    errors = []
    tasks = registry.get("tasks", [])
    sources = registry.get("sources", [])
    models = registry.get("models", [])
    source_map = {row.get("source_id"): row for row in sources}

    def unique(rows, key, label):
        values = [row.get(key) for row in rows]
        if None in values or "" in values:
            errors.append(f"{label} has empty {key}")
        if len(values) != len(set(values)):
            errors.append(f"{label} has duplicate {key}")

    unique(tasks, "task_id", "tasks")
    unique(tasks, "slug", "tasks")
    unique(sources, "source_id", "sources")
    unique(models, "model_id", "models")
    if registry.get("frozen_task_count") != len(tasks):
        errors.append("frozen_task_count mismatch")
    if registry.get("policy", {}).get("organism_scope") != "plant_or_crop_only":
        errors.append("organism policy is not plant/crop only")
    policy = registry.get("policy", {})
    if policy.get("license_execution_policy") != LICENSE_EXECUTION_POLICY:
        errors.append("license execution policy is not user-authorized direct use")
    if policy.get("license_metadata_is_execution_gate") is not False:
        errors.append("license metadata must not be an execution gate")
    if policy.get("internal_pretraining_ablations") != INTERNAL_PRETRAINING_ABLATIONS:
        errors.append("internal pretraining ablations are not disabled by user")

    serialized = json.dumps(registry, ensure_ascii=False).lower()
    for term in FORBIDDEN_SCOPE_TERMS:
        if term in serialized:
            errors.append(f"forbidden non-plant scope term: {term}")

    required_task_fields = (
        "task_id", "slug", "title_cn", "suite", "organism_scope", "model_input",
        "task_kind", "primary_metric", "source_id", "adapter", "split_policy",
        "data_state", "formal_state", "execution_state", "biological_question",
    )
    for row in tasks:
        task_id = row.get("task_id", "<unknown>")
        for field in required_task_fields:
            if row.get(field) in (None, ""):
                errors.append(f"{task_id}: missing {field}")
        if row.get("organism_scope") != "plant_or_crop":
            errors.append(f"{task_id}: non-plant organism_scope")
        if row.get("model_input") != "dna_sequence_only":
            errors.append(f"{task_id}: main input is not DNA-only")
        if row.get("adapter") not in ALLOWED_ADAPTERS:
            errors.append(f"{task_id}: unsupported adapter {row.get('adapter')!r}")
        if row.get("execution_state") not in ALLOWED_EXECUTION_STATES:
            errors.append(f"{task_id}: unsupported execution_state")
        source_id = row.get("source_id")
        if source_id not in source_map:
            errors.append(f"{task_id}: unknown source {source_id!r}")

    for row in sources:
        source_id = row.get("source_id", "<unknown>")
        if row.get("kind") in {"huggingface_dataset", "git_repository"} and not row.get("revision"):
            errors.append(f"{source_id}: remote source is not revision pinned")
        if row.get("kind") == "zenodo" and not row.get("record_id"):
            errors.append(f"{source_id}: Zenodo source lacks record_id")
        if not str(row.get("url", "")).startswith(("https://", "file://")):
            errors.append(f"{source_id}: invalid source URL")

    for row in models:
        model_id = row.get("model_id", "<unknown>")
        if model_id in FORBIDDEN_INTERNAL_MODEL_IDS or row.get("kind") == "internal_control":
            errors.append(f"{model_id}: internal pretraining ablation is disabled by user")
        if not row.get("capabilities"):
            errors.append(f"{model_id}: no capabilities")
        if row.get("kind") == "public_weight":
            if not row.get("revision"):
                errors.append(f"{model_id}: public weight is not revision pinned")

    report = {
        "status": "ok" if not errors else "invalid",
        "registry_version": registry.get("registry_version"),
        "task_count": len(tasks),
        "source_count": len(sources),
        "model_count": len(models),
        "errors": errors,
    }
    if errors and raise_on_error:
        raise ValueError("invalid downstream v4 registry: " + "; ".join(errors))
    return report
