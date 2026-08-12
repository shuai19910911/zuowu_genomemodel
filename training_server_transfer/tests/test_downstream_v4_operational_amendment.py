import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.operational_amendment import (
    ALLOWED_MEMORY_OPTIMIZATION_ASSETS,
    PREEXISTING_DRIFT_ASSETS,
    valid_memory_optimization_amendment,
    write_memory_optimization_amendment,
)
from downstream_v4.formal_gate import _implementation_drift_evidence


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def test_memory_optimization_amendment_binds_exact_drift_and_evidence(tmp_path):
    project_root = tmp_path / "project"
    final_root = tmp_path / "final"
    final_root.mkdir(parents=True)
    old_assets = {}
    live_assets = {}
    all_drift = ALLOWED_MEMORY_OPTIMIZATION_ASSETS | PREEXISTING_DRIFT_ASSETS
    for index, relative in enumerate(sorted(all_drift)):
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        current = ("current-%d" % index).encode()
        path.write_bytes(current)
        if relative in PREEXISTING_DRIFT_ASSETS:
            path.touch()
            import os
            os.utime(path, (1, 1))
        old_assets[relative] = {
            "sha256": _sha(("old-%d" % index).encode()), "size_bytes": index + 1,
        }
        live_assets[relative] = {
            "sha256": _sha(current), "size_bytes": len(current),
        }
    frozen = {"plan_sha256": "frozen-plan", "implementation_assets": old_assets}
    live = {"plan_sha256": "live-plan", "implementation_assets": live_assets}

    cpu = final_root / "controller" / "MEMORY_OPTIMIZATION_CPU_TESTS.json"
    memory = final_root / "controller" / "MEMORY_OPTIMIZATION_BENCHMARK.json"
    gpu = final_root / "controller" / "MEMORY_OPTIMIZATION_GPU_EQUIVALENCE.json"
    smoke = final_root / "controller" / "MEMORY_OPTIMIZATION_SMOKE_MIGRATION.json"
    cpu.parent.mkdir(parents=True)
    cpu.write_text(json.dumps({"status": "ok", "tests_passed": 203}) + "\n")
    memory.write_text(json.dumps({"status": "ok", "rss_reduction_percent": 73.38}) + "\n")
    gpu.write_text(json.dumps({
        "status": "ok", "equivalence_gate": {"passed": True},
    }) + "\n")
    smoke.write_text(json.dumps({
        "status": "ok", "models_total": 14,
    }) + "\n")

    amendment = write_memory_optimization_amendment(
        final_root, frozen, live, project_root, [cpu, memory, gpu, smoke],
    )
    assert amendment["status"] == "active"
    assert valid_memory_optimization_amendment(
        final_root, frozen, live, project_root,
    ) is not None
    bound_evidence = _implementation_drift_evidence(
        frozen, live, project_root, final_root,
    )
    assert bound_evidence is not None
    assert {path.name for path in bound_evidence} == {
        "MEMORY_OPTIMIZATION_IMPLEMENTATION_AMENDMENT.json",
        "MEMORY_OPTIMIZATION_CPU_TESTS.json",
        "MEMORY_OPTIMIZATION_BENCHMARK.json",
        "MEMORY_OPTIMIZATION_GPU_EQUIVALENCE.json",
        "MEMORY_OPTIMIZATION_SMOKE_MIGRATION.json",
    }

    gpu.write_text(json.dumps({
        "status": "failed", "equivalence_gate": {"passed": False},
    }) + "\n")
    assert valid_memory_optimization_amendment(
        final_root, frozen, live, project_root,
    ) is None
    assert _implementation_drift_evidence(
        frozen, live, project_root, final_root,
    ) is None
