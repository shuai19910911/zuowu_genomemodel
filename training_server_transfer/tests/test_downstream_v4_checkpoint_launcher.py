import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.checkpoint_launcher import build_launch_candidate


def test_checkpoint_launcher_builds_single_checkpoint_article_plan(tmp_path):
    checkpoint = tmp_path / "step_00012345.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    final_root = tmp_path / "run"
    final_root.mkdir()
    candidate = build_launch_candidate(
        project_root=ROOT,
        final_root=final_root,
        checkpoint_path=checkpoint,
        registry_path=ROOT / "training_server_transfer/configs/cropgenome_downstream_v4.json",
        profile_path=ROOT / "training_server_transfer/configs/downstream_article_guided_v1.json",
        model_config_path=ROOT / "training_server_transfer/configs/model_large.json",
        scheduler_template_path=(
            ROOT / "training_server_transfer/configs/downstream_article_gpu05_2080ti.json"
        ),
    )
    plan = candidate["plan"]
    assert plan["checkpoint_schedule"]["steps"] == [12345]
    assert list(plan["checkpoint_identities"]) == ["step_00012345"]
    assert Path(plan["checkpoint_identities"]["step_00012345"]["path"]) == checkpoint
    assert plan["task_selection"]["task_ids"]
    assert len(set(plan["task_selection"]["task_ids"])) == 25
    assert candidate["launch_contract"]["max_workers"] == 3
    assert candidate["launch_contract"]["max_tasks_per_gpu"] == 1
    assert candidate["launch_contract"]["foreign_compute_allowed"] is False
    assert "daemon" in candidate["remote_command"]
    assert "--max-workers 3" in candidate["remote_command"]
