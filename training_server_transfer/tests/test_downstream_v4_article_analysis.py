import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.article_analysis import build_article_summary


def test_article_summary_reports_macro_micro_specialization_and_difficulty():
    plan = {
        "plan_sha256": "p" * 64,
        "protocol_id": "cropgenome_downstream_article_guided_checkpoint_v1",
        "rows": [
            {"run_key": "m1_t1", "model_id": "m1", "task_id": "t1", "task_kind": "binary_classification", "article_category": "cat_a", "context_bp": 512, "checkpoint_step": None},
            {"run_key": "m1_t2", "model_id": "m1", "task_id": "t2", "task_kind": "binary_classification", "article_category": "cat_b", "context_bp": 512, "checkpoint_step": None},
            {"run_key": "m2_t1", "model_id": "m2", "task_id": "t1", "task_kind": "binary_classification", "article_category": "cat_a", "context_bp": 512, "checkpoint_step": None},
            {"run_key": "m2_t2", "model_id": "m2", "task_id": "t2", "task_kind": "binary_classification", "article_category": "cat_b", "context_bp": 512, "checkpoint_step": None},
        ],
    }
    receipts = [
        {"run_key": "m1_t1", "test_metrics": {"mcc": 0.9}},
        {"run_key": "m1_t2", "test_metrics": {"mcc": 0.1}},
        {"run_key": "m2_t1", "test_metrics": {"mcc": 0.3}},
        {"run_key": "m2_t2", "test_metrics": {"mcc": 0.1}},
    ]
    result = build_article_summary(plan, receipts)
    assert result["status"] == "complete"
    by_model = {row["model_key"]: row for row in result["model_context_summary"]}
    assert by_model["m1"]["macro_category_mcc"] == 0.5
    assert by_model["m1"]["micro_task_mcc"] == 0.5
    specialization = {
        (row["model_key"], row["category"]): row["specialization_score"]
        for row in result["specialization"]
    }
    assert specialization[("m1", "cat_a")] > 0
    assert specialization[("m1", "cat_b")] < 0
    assert [row["task_id"] for row in result["hard_tasks"]] == ["t2"]
    assert [row["task_id"] for row in result["high_variance_tasks"]] == ["t1"]
