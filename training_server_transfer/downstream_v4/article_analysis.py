"""Category-aware GENEB-style summaries for article-guided results."""

from collections import defaultdict

import numpy as np
from scipy.stats import rankdata


ARTICLE_ANALYSIS_PROTOCOL_ID = "cropgenome_article_category_analysis_v1"


def _model_key(row):
    step = row.get("checkpoint_step")
    if step is None:
        return str(row["model_id"])
    return f"{row['model_id']}@step{int(step):08d}"


def _classification_rows(plan):
    return [
        row for row in plan.get("rows") or []
        if row.get("task_kind") in {"binary_classification", "multiclass_classification"}
        and row.get("execution_kind", "evaluation") == "evaluation"
        and row.get("article_category")
    ]


def build_article_summary(plan, result_receipts):
    rows = _classification_rows(plan)
    by_run_key = {row["run_key"]: row for row in rows}
    observed = {}
    for receipt in result_receipts:
        run_key = str(receipt.get("run_key", ""))
        metrics = receipt.get("test_metrics") or {}
        if run_key in by_run_key and isinstance(metrics.get("mcc"), (int, float)):
            observed[run_key] = float(metrics["mcc"])
    expected = set(by_run_key)
    missing = sorted(expected - set(observed))
    status = "complete" if not missing else "incomplete"

    records = []
    for run_key, value in sorted(observed.items()):
        row = by_run_key[run_key]
        records.append({
            "run_key": run_key,
            "model_key": _model_key(row),
            "model_id": row["model_id"],
            "checkpoint_step": row.get("checkpoint_step"),
            "context_bp": int(row["context_bp"]),
            "task_id": row["task_id"],
            "category": row["article_category"],
            "mcc": value,
        })

    grouped = defaultdict(list)
    for record in records:
        grouped[(record["model_key"], record["context_bp"])].append(record)
    model_context_summary = []
    for (model_key, context), values in sorted(grouped.items()):
        category_values = defaultdict(list)
        for record in values:
            category_values[record["category"]].append(record["mcc"])
        category_mcc = {
            category: float(np.mean(scores))
            for category, scores in sorted(category_values.items())
        }
        model_context_summary.append({
            "model_key": model_key,
            "context_bp": int(context),
            "category_mcc": category_mcc,
            "macro_category_mcc": float(np.mean(list(category_mcc.values()))),
            "micro_task_mcc": float(np.mean([record["mcc"] for record in values])),
            "tasks": int(len(values)),
            "categories": int(len(category_mcc)),
        })

    task_groups = defaultdict(list)
    for record in records:
        task_groups[(record["task_id"], record["context_bp"])].append(record)
    task_difficulty = []
    for (task_id, context), values in sorted(task_groups.items()):
        scores = np.asarray([record["mcc"] for record in values], dtype=float)
        task_difficulty.append({
            "task_id": task_id,
            "context_bp": int(context),
            "mean_mcc": float(scores.mean()),
            "std_mcc": float(scores.std(ddof=0)),
            "models": int(len(scores)),
        })
    hard_tasks = [
        row for row in task_difficulty if row["mean_mcc"] < 0.35
    ]
    high_variance_tasks = [
        row for row in task_difficulty if row["std_mcc"] > 0.12
    ]

    ranks = {}
    for (task_id, context), values in sorted(task_groups.items()):
        ordered = sorted(values, key=lambda row: row["model_key"])
        task_ranks = rankdata(-np.asarray([row["mcc"] for row in ordered]), method="average")
        for record, rank in zip(ordered, task_ranks):
            ranks[(record["model_key"], int(context), task_id)] = float(rank)
    specialization = []
    for (model_key, context), values in sorted(grouped.items()):
        model_task_categories = {
            record["task_id"]: record["category"] for record in values
            if (model_key, int(context), record["task_id"]) in ranks
        }
        all_tasks = sorted(model_task_categories)
        for category in sorted(set(model_task_categories.values())):
            inside = [
                ranks[(model_key, int(context), task_id)]
                for task_id in all_tasks if model_task_categories[task_id] == category
            ]
            outside = [
                ranks[(model_key, int(context), task_id)]
                for task_id in all_tasks if model_task_categories[task_id] != category
            ]
            if not inside or not outside:
                continue
            specialization.append({
                "model_key": model_key,
                "context_bp": int(context),
                "category": category,
                "within_category_mean_rank": float(np.mean(inside)),
                "outside_category_mean_rank": float(np.mean(outside)),
                "specialization_score": float(np.mean(outside) - np.mean(inside)),
            })

    return {
        "protocol_id": ARTICLE_ANALYSIS_PROTOCOL_ID,
        "status": status,
        "plan_sha256": plan.get("plan_sha256"),
        "expected_classification_rows": int(len(expected)),
        "observed_classification_rows": int(len(observed)),
        "missing_run_keys": missing,
        "model_context_summary": model_context_summary,
        "task_difficulty": task_difficulty,
        "hard_tasks": hard_tasks,
        "high_variance_tasks": high_variance_tasks,
        "specialization": specialization,
        "final_leaderboard_allowed": status == "complete",
    }
