"""Hash-bound operational amendment for numerically equivalent memory optimization."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


AMENDMENT_FILENAME = "controller/MEMORY_OPTIMIZATION_IMPLEMENTATION_AMENDMENT.json"
AMENDMENT_PROTOCOL_ID = "cropgenome_downstream_memory_optimization_equivalence_v1"
ALLOWED_MEMORY_OPTIMIZATION_ASSETS = frozenset({
    "scripts/extract_cropgenome_bench_v1_embeddings.py",
    "scripts/extract_public_dna_embeddings.py",
    "scripts/run_cropgenome_downstream_final.py",
    "training_server_transfer/downstream_v4/commands.py",
    "training_server_transfer/downstream_v4/final_audit.py",
    "training_server_transfer/downstream_v4/final_controller.py",
    "training_server_transfer/downstream_v4/final_protocol.py",
    "training_server_transfer/downstream_v4/formal_gate.py",
    "training_server_transfer/downstream_v4/gpu_gate.py",
    "training_server_transfer/downstream_v4/operational_amendment.py",
    "training_server_transfer/downstream_v4/streaming_embeddings.py",
})
PREEXISTING_DRIFT_ASSETS = frozenset({
    "training_server_transfer/downstream_v4/cpu_scheduler.py",
    "training_server_transfer/scripts/train.py",
})
MEMORY_OPTIMIZATION_STARTED_AT = "2026-08-11T13:09:47+00:00"
MEMORY_OPTIMIZATION_STARTED_NS = int(
    datetime.fromisoformat(MEMORY_OPTIMIZATION_STARTED_AT).timestamp() * 1_000_000_000
)
EXPECTED_EVIDENCE = {
    "MEMORY_OPTIMIZATION_CPU_TESTS.json",
    "MEMORY_OPTIMIZATION_BENCHMARK.json",
    "MEMORY_OPTIMIZATION_GPU_EQUIVALENCE.json",
    "MEMORY_OPTIMIZATION_SMOKE_MIGRATION.json",
}


def _canonical_sha(payload):
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _asset_map(plan):
    assets = plan.get("implementation_assets") or {}
    if isinstance(assets, dict):
        return {
            str(path): str(record["sha256"] if isinstance(record, dict) else record)
            for path, record in assets.items()
        }
    return {
        str(record["path"]): str(record["sha256"])
        for record in assets
    }


def implementation_drift_records(frozen_plan, live_plan):
    frozen = _asset_map(frozen_plan)
    live = _asset_map(live_plan)
    return [
        {
            "path": path,
            "frozen_sha256": frozen.get(path),
            "live_sha256": live.get(path),
        }
        for path in sorted(set(frozen) | set(live))
        if frozen.get(path) != live.get(path)
    ]


def _evidence_semantics(records):
    by_name = {Path(record["path"]).name: record for record in records}
    if set(by_name) != EXPECTED_EVIDENCE:
        return False
    try:
        cpu = json.loads(Path(by_name["MEMORY_OPTIMIZATION_CPU_TESTS.json"]["path"]).read_text())
        memory = json.loads(Path(by_name["MEMORY_OPTIMIZATION_BENCHMARK.json"]["path"]).read_text())
        gpu = json.loads(Path(by_name["MEMORY_OPTIMIZATION_GPU_EQUIVALENCE.json"]["path"]).read_text())
        smoke = json.loads(Path(by_name["MEMORY_OPTIMIZATION_SMOKE_MIGRATION.json"]["path"]).read_text())
    except (OSError, ValueError, KeyError):
        return False
    return bool(
        cpu.get("status") == "ok"
        and int(cpu.get("tests_passed", 0)) >= 200
        and memory.get("status") == "ok"
        and float(memory.get("rss_reduction_percent", 0.0)) >= 50.0
        and gpu.get("status") == "ok"
        and (gpu.get("equivalence_gate") or {}).get("passed") is True
        and smoke.get("status") == "ok"
        and int(smoke.get("models_total", 0)) == 14
    )


def _evidence_records(paths):
    records = []
    for raw in sorted({str(Path(path).resolve()) for path in paths}):
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_path(path),
        })
    if not _evidence_semantics(records):
        raise RuntimeError("memory optimization evidence is incomplete or failed")
    return records


def write_memory_optimization_amendment(
    final_root, frozen_plan, live_plan, project_root, evidence_paths,
):
    drift = implementation_drift_records(frozen_plan, live_plan)
    expected_paths = ALLOWED_MEMORY_OPTIMIZATION_ASSETS | PREEXISTING_DRIFT_ASSETS
    if {record["path"] for record in drift} != expected_paths:
        raise RuntimeError("implementation drift is outside the memory optimization whitelist")
    project_root = Path(project_root).resolve()
    live_assets = _asset_map(live_plan)
    for relative in expected_paths:
        path = project_root / relative
        if not path.is_file() or _sha256_path(path) != live_assets.get(relative):
            raise RuntimeError("live implementation asset mismatch: " + relative)
    preexisting = []
    for record in drift:
        if record["path"] not in PREEXISTING_DRIFT_ASSETS:
            continue
        stat = (project_root / record["path"]).stat()
        if stat.st_mtime_ns >= MEMORY_OPTIMIZATION_STARTED_NS:
            raise RuntimeError("pre-existing drift was modified during memory optimization")
        preexisting.append({**record, "mtime_ns": stat.st_mtime_ns})
    evidence = _evidence_records(evidence_paths)
    payload = {
        "status": "active",
        "protocol_id": AMENDMENT_PROTOCOL_ID,
        "frozen_plan_sha256": frozen_plan.get("plan_sha256"),
        "live_implementation_plan_sha256": live_plan.get("plan_sha256"),
        "memory_optimization_assets": [
            record for record in drift
            if record["path"] in ALLOWED_MEMORY_OPTIMIZATION_ASSETS
        ],
        "preexisting_drift_assets": preexisting,
        "memory_optimization_started_at": MEMORY_OPTIMIZATION_STARTED_AT,
        "supporting_evidence": evidence,
        "user_authorization": "optimize_downstream_and_cancel_running_2080_programs",
        "scientific_scope_unchanged": True,
        "cache_field_schema_unchanged": True,
        "metric_and_split_selection_unchanged": True,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    payload["receipt_sha256"] = _canonical_sha(payload)
    path = Path(final_root).resolve() / AMENDMENT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))
    return payload


def valid_memory_optimization_amendment(final_root, frozen_plan, live_plan, project_root):
    path = Path(final_root).resolve() / AMENDMENT_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    canonical = dict(payload)
    stored = canonical.pop("receipt_sha256", None)
    expected_drift = implementation_drift_records(frozen_plan, live_plan)
    expected_memory = [
        record for record in expected_drift
        if record["path"] in ALLOWED_MEMORY_OPTIMIZATION_ASSETS
    ]
    expected_preexisting = []
    project_root = Path(project_root).resolve()
    for record in expected_drift:
        if record["path"] not in PREEXISTING_DRIFT_ASSETS:
            continue
        path_stat = (project_root / record["path"]).stat()
        expected_preexisting.append({**record, "mtime_ns": path_stat.st_mtime_ns})
    if not (
        payload.get("status") == "active"
        and payload.get("protocol_id") == AMENDMENT_PROTOCOL_ID
        and payload.get("frozen_plan_sha256") == frozen_plan.get("plan_sha256")
        and payload.get("live_implementation_plan_sha256") == live_plan.get("plan_sha256")
        and payload.get("memory_optimization_assets") == expected_memory
        and payload.get("preexisting_drift_assets") == expected_preexisting
        and {record["path"] for record in expected_drift}
        == ALLOWED_MEMORY_OPTIMIZATION_ASSETS | PREEXISTING_DRIFT_ASSETS
        and payload.get("memory_optimization_started_at") == MEMORY_OPTIMIZATION_STARTED_AT
        and all(
            int(record["mtime_ns"]) < MEMORY_OPTIMIZATION_STARTED_NS
            for record in expected_preexisting
        )
        and payload.get("user_authorization")
        == "optimize_downstream_and_cancel_running_2080_programs"
        and payload.get("scientific_scope_unchanged") is True
        and payload.get("cache_field_schema_unchanged") is True
        and payload.get("metric_and_split_selection_unchanged") is True
        and stored == _canonical_sha(canonical)
    ):
        return None
    live_assets = _asset_map(live_plan)
    for relative in ALLOWED_MEMORY_OPTIMIZATION_ASSETS | PREEXISTING_DRIFT_ASSETS:
        current = project_root / relative
        if not current.is_file() or _sha256_path(current) != live_assets.get(relative):
            return None
    evidence = payload.get("supporting_evidence") or []
    for record in evidence:
        evidence_path = Path(record.get("path", ""))
        if (
            not evidence_path.is_file()
            or evidence_path.stat().st_size != int(record.get("size_bytes", -1))
            or _sha256_path(evidence_path) != record.get("sha256")
        ):
            return None
    if not _evidence_semantics(evidence):
        return None
    return payload
