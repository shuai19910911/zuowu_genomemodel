import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_cropgenome_downstream_final.py"
SPEC = importlib.util.spec_from_file_location("cropgenome_final_cli", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_daemon_and_group_defaults_defer_to_frozen_gpu_policy():
    daemon = MODULE.build_parser().parse_args(["daemon"])
    group = MODULE.build_parser().parse_args(["run-group", "--group-key", "g"])
    assert daemon.max_workers == 0
    assert daemon.max_tasks_per_gpu is None
    assert daemon.allow_foreign_compute is None
    assert group.max_tasks_per_gpu is None
    assert group.allow_foreign_compute is None
