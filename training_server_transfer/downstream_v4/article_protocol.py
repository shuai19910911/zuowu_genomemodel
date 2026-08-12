"""Article-guided, plant-only downstream profile for an explicit checkpoint."""

import copy
import hashlib
import json
from pathlib import Path

from .latest_snapshot import build_checkpoint_set_plan
from .probe import ARTICLE_PROBE_PROTOCOL_ID, PROBE_PROTOCOL_ID


ARTICLE_GUIDED_PROTOCOL_ID = "cropgenome_downstream_article_guided_checkpoint_v1"
ARTICLE_PROFILE_PROTOCOL_ID = "cropgenome_downstream_article_guided_v1"


def _canonical_sha(payload):
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_article_profile(path):
    path = Path(path).resolve()
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("protocol_id") != ARTICLE_PROFILE_PROTOCOL_ID:
        raise RuntimeError("unexpected article-guided profile protocol")
    core = list(map(str, profile.get("core_task_ids") or []))
    supplement = list(map(str, profile.get("supplement_task_ids") or []))
    selected = core + supplement
    if not selected or len(selected) != len(set(selected)):
        raise RuntimeError("article-guided task IDs must be non-empty and unique")
    if int(profile.get("complete_task_count", -1)) != len(selected):
        raise RuntimeError("article-guided complete_task_count mismatch")
    categories = profile.get("category_by_task") or {}
    if set(categories) != set(selected):
        raise RuntimeError("every article-guided task requires exactly one category")
    seeds = (profile.get("probe_policy") or {}).get("seeds") or []
    if list(map(int, seeds)) != [13, 17, 42, 123, 997]:
        raise RuntimeError("article-guided profile requires fixed GENEB five seeds")
    if profile.get("scope") != "plant_and_crop_dna_only":
        raise RuntimeError("article-guided profile must remain plant/crop only")
    return profile


def _filtered_registry(registry, selected_task_ids):
    selected = set(map(str, selected_task_ids))
    available = {task["task_id"]: task for task in registry["tasks"]}
    missing = sorted(selected - set(available))
    if missing:
        raise RuntimeError("article-guided registry tasks missing: " + ", ".join(missing))
    tasks = [copy.deepcopy(available[task_id]) for task_id in sorted(selected)]
    invalid = [
        task["task_id"] for task in tasks
        if task.get("organism_scope") != "plant_or_crop"
        or task.get("model_input") != "dna_sequence_only"
    ]
    if invalid:
        raise RuntimeError("article-guided tasks violate plant DNA-only scope: " + ", ".join(invalid))
    filtered = copy.deepcopy(registry)
    filtered["tasks"] = tasks
    return filtered


def build_article_guided_checkpoint_plan(
    registry, checkpoint_identities, project_model_config,
    gpu_scheduler_policy, profile,
):
    profile = copy.deepcopy(profile)
    selected = list(map(str, profile["core_task_ids"] + profile["supplement_task_ids"]))
    filtered = _filtered_registry(registry, selected)
    plan = build_checkpoint_set_plan(
        filtered,
        checkpoint_identities=checkpoint_identities,
        project_model_config=project_model_config,
        gpu_scheduler_policy=gpu_scheduler_policy,
        excluded_task_ids=(),
    )
    plan.pop("plan_sha256", None)
    seeds = [int(value) for value in profile["probe_policy"]["seeds"]]
    few_shot_tasks = set(map(str, profile["probe_policy"]["few_shot_task_ids"]))
    categories = profile["category_by_task"]
    for row in plan["rows"]:
        row["seeds"] = list(seeds)
        row["article_category"] = categories[row["task_id"]]
        row["classification_reporting_metrics"] = (
            ["mcc", row["primary_metric"]]
            if "classification" in row["task_kind"]
            else [row["primary_metric"]]
        )
        row["few_shot_regimes"] = (
            list(map(int, profile["probe_policy"]["few_shot_regimes"]))
            if row["task_id"] in few_shot_tasks
            and row["task_kind"] in {"binary_classification", "multiclass_classification"}
            else []
        )
        row["training_row_cap"] = (
            int(profile["resource_policy"]["training_row_cap"])
            if row["cv_policy"] == "fixed_train_validation_test"
            and row["task_kind"] in {
                "binary_classification", "multiclass_classification",
                "multilabel_classification", "regression", "multioutput_regression",
            }
            else None
        )
        if row["evaluation_protocol_id"] == PROBE_PROTOCOL_ID:
            row["evaluation_protocol_id"] = ARTICLE_PROBE_PROTOCOL_ID
    plan["protocol_id"] = ARTICLE_GUIDED_PROTOCOL_ID
    plan["profile"] = "article_guided_plant_only"
    plan["seeds"] = list(seeds)
    plan["task_ids"] = sorted(selected)
    plan["excluded_task_ids"] = []
    plan["task_selection"] = {
        "selection": "explicit_article_guided_profile",
        "task_ids": sorted(selected),
        "core_task_ids": list(map(str, profile["core_task_ids"])),
        "supplement_task_ids": list(map(str, profile["supplement_task_ids"])),
    }
    plan["article_guided_evaluation"] = profile
    plan["probe_protocols"]["pooled"] = ARTICLE_PROBE_PROTOCOL_ID
    plan["article_guided_profile_sha256"] = _canonical_sha(profile)
    policy = dict(plan["gpu_scheduler_policy"])
    policy["max_workers"] = int(profile["resource_policy"]["default_max_workers"])
    policy["host_reserved_memory_mib"] = int(
        profile["resource_policy"]["host_memory_reserved_mib"]
    )
    policy["max_tasks_per_gpu"] = 1
    policy["foreign_compute_allowed"] = False
    policy["unknown_compute_blocks"] = True
    plan["gpu_scheduler_policy"] = policy
    plan["test_policy"] = {
        **plan["test_policy"],
        "checkpoint_selection_rule": "explicit_user_supplied_checkpoint",
        "checkpoint_trajectory_is_monitoring_evidence": True,
        "metrics_sealed_until_matrix_complete": True,
        "article_guided_profile_required": True,
        "validation_and_test_never_subsampled": True,
    }
    plan["plan_sha256"] = _canonical_sha(plan)
    return plan
