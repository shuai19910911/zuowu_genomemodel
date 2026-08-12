"""Prepare and index the one final canonical dataset farm."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .data import sha256_path
from .final_protocol import EXCLUDED_TASK_IDS
from .preparation import EXISTING_DATASETS, prepare_task


def _write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _valid_materialized_dataset(root, task_id):
    root = Path(root).resolve(); manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("status") != "ok" or manifest.get("task_id") != task_id:
        return None
    if (
        manifest.get("schema_version") != "canonical-sequence-v4.1"
        or manifest.get("implementation_sha256") != sha256_path(Path(__file__).with_name("data.py"))
        or (manifest.get("source_receipt") or {}).get("preparation_implementation_sha256")
        != sha256_path(Path(__file__).with_name("preparation.py"))
        or (manifest.get("source_receipt") or {}).get("adapter_implementation_sha256")
        != sha256_path(Path(__file__).with_name("adapters.py"))
    ):
        return None
    for relative, record in (manifest.get("artifacts") or {}).items():
        artifact = root / relative
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(record.get("size_bytes", -1))
            or sha256_path(artifact) != record.get("sha256")
        ):
            return None
    return {
        "status": "ok", "task_id": task_id, "dataset_root": str(root),
        "profile": "full", "manifest": manifest,
        "manifest_path": str(manifest_path), "manifest_sha256": sha256_path(manifest_path),
    }


def _valid_dataset_farm_entry(task_id, receipt_path, farm):
    receipt_path = Path(receipt_path)
    farm = Path(farm)
    if not receipt_path.is_file() or not farm.is_symlink() or not (farm / "samples.tsv").is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    target = farm.resolve()
    if receipt.get("task_id") != task_id or Path(receipt.get("dataset_root", "")).resolve() != target:
        return None
    if receipt.get("status") == "adopted" and receipt.get("profile") == "existing_frozen":
        manifest_path = Path(receipt.get("manifest_path", ""))
        samples_path = target / "samples.tsv"
        if (
            not manifest_path.is_file()
            or sha256_path(manifest_path) != receipt.get("manifest_sha256")
            or sha256_path(samples_path) != receipt.get("samples_sha256")
        ):
            return None
        return {"profile": "existing_frozen", "manifest_sha256": receipt["manifest_sha256"]}
    valid = _valid_materialized_dataset(target, task_id)
    if (
        valid is None
        or receipt.get("status") != "ok"
        or receipt.get("manifest_sha256") != valid["manifest_sha256"]
        or Path(receipt.get("manifest_path", "")).resolve() != Path(valid["manifest_path"]).resolve()
    ):
        return None
    return {"profile": "canonical_v4_1", "manifest_sha256": valid["manifest_sha256"]}


def _bind_link(link, target):
    link = Path(link); target = Path(target).resolve(); link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"dataset farm path is not a symlink: {link}")
    link.symlink_to(target, target_is_directory=True)


def prepare_final_task(registry, task_id, project_root, final_root):
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    task = next((row for row in registry["tasks"] if row["task_id"] == task_id), None)
    if task is None:
        raise KeyError(task_id)
    if task_id in EXCLUDED_TASK_IDS or task_id == "B17":
        raise ValueError(f"task is not materialized by the final data stage: {task_id}")
    if task_id in EXISTING_DATASETS:
        receipt = prepare_task(registry, task_id, project_root, profile="full")
    else:
        destination = final_root / "prepared_datasets" / task_id
        receipt = _valid_materialized_dataset(destination, task_id)
        if receipt is None:
            receipt = prepare_task(
                registry, task_id, project_root, profile="full", output_root=destination,
            )
    dataset_root = Path(receipt["dataset_root"]).resolve()
    _bind_link(final_root / "datasets" / task_id, dataset_root)
    receipt_path = final_root / "dataset_receipts" / f"{task_id}.json"
    _write_json(receipt_path, receipt)
    return {
        "task_id": task_id, "status": "ready", "dataset_root": str(dataset_root),
        "farm_path": str((final_root / "datasets" / task_id).absolute()),
        "receipt_path": str(receipt_path), "receipt_sha256": sha256_path(receipt_path),
    }


def refresh_dataset_index(registry, final_root):
    final_root = Path(final_root).resolve(); rows = []
    for task in registry["tasks"]:
        task_id = task["task_id"]
        if task_id in EXCLUDED_TASK_IDS:
            rows.append({"task_id": task_id, "status": "excluded_edta"}); continue
        if task_id == "B17":
            rows.append({"task_id": task_id, "status": "analysis_only"}); continue
        receipt_path = final_root / "dataset_receipts" / f"{task_id}.json"
        farm = final_root / "datasets" / task_id
        validation = _valid_dataset_farm_entry(task_id, receipt_path, farm)
        if validation is not None:
            rows.append({
                "task_id": task_id, "status": "ready", "dataset_root": str(farm.resolve()),
                "farm_path": str(farm.absolute()), "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_path(receipt_path),
                **validation,
            })
        else:
            rows.append({"task_id": task_id, "status": "waiting"})
    index = {
        "status": "ready" if all(row["status"] in {"ready", "excluded_edta", "analysis_only"} for row in rows) else "waiting",
        "profile": "full", "rows": rows,
        "ready": sum(row["status"] == "ready" for row in rows),
        "waiting": sum(row["status"] == "waiting" for row in rows),
        "farm_root": str(final_root / "datasets"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(final_root / "DATASET_INDEX.json", index)
    return index


def prepare_final_datasets(registry, project_root, final_root):
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    prepared_root = final_root / "prepared_datasets"
    farm_root = final_root / "datasets"
    receipt_root = final_root / "dataset_receipts"
    rows = []
    for task in registry["tasks"]:
        task_id = task["task_id"]
        if task_id in EXCLUDED_TASK_IDS:
            rows.append({"task_id": task_id, "status": "excluded_edta"}); continue
        if task_id == "B17":
            rows.append({"task_id": task_id, "status": "analysis_only"}); continue
        try:
            rows.append(prepare_final_task(registry, task_id, project_root, final_root))
        except Exception as error:
            rows.append({"task_id": task_id, "status": "waiting", "error": repr(error)})
    index = {
        "status": "ready" if all(row["status"] in {"ready", "excluded_edta", "analysis_only"} for row in rows) else "waiting",
        "profile": "full", "rows": rows,
        "ready": sum(row["status"] == "ready" for row in rows),
        "waiting": sum(row["status"] == "waiting" for row in rows),
        "farm_root": str(farm_root),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(final_root / "DATASET_INDEX.json", index)
    return index
