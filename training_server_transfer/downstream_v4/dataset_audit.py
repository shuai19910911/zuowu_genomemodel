"""Streaming integrity audit for the final canonical dataset farm."""

import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .data import sha256_path
from .final_protocol import EXCLUDED_TASK_IDS


DATASET_AUDIT_PROTOCOL_ID = "canonical_split_integrity_sqlite_v1"
_SPLIT_ALIASES = {"val": "validation", "valid": "validation", "dev": "validation"}


def _cv_policy(task):
    if str(task.get("task_kind") or "").startswith("zero_shot"):
        return "official_test_only_zero_shot"
    if task.get("split_policy") == "official_leave_one_chromosome_out":
        return "leave_one_group_out"
    return "fixed_train_validation_test"


def _canonical_sha(payload):
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _valid_existing_task_audit(task, final_root):
    """Reuse a complete row audit only while its immutable hash chain still matches."""
    final_root = Path(final_root).resolve()
    task_id = task["task_id"]
    path = final_root / "dataset_audits" / f"{task_id}.json"
    dataset_receipt_path = final_root / "dataset_receipts" / f"{task_id}.json"
    samples = final_root / "datasets" / task_id / "samples.tsv"
    if not path.is_file() or not dataset_receipt_path.is_file() or not samples.is_file():
        return None
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
        canonical = dict(audit)
        stored_sha256 = canonical.pop("receipt_sha256", None)
        dataset_receipt = json.loads(dataset_receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    cv_policy = _cv_policy(task)
    if (
        stored_sha256 != _canonical_sha(canonical)
        or audit.get("status") != "ok"
        or audit.get("protocol_id") != DATASET_AUDIT_PROTOCOL_ID
        or audit.get("task_id") != task_id
        or audit.get("task_kind") != task["task_kind"]
        or audit.get("split_policy") != task.get("split_policy")
        or audit.get("cv_policy") != cv_policy
        or Path(audit.get("samples_path", "")).resolve() != samples.resolve()
        or Path(audit.get("dataset_receipt_path", "")).resolve()
        != dataset_receipt_path.resolve()
        or audit.get("dataset_receipt_sha256") != sha256_path(dataset_receipt_path)
        or dataset_receipt.get("task_id") != task_id
        or Path(dataset_receipt.get("dataset_root", "")).resolve() != samples.parent.resolve()
    ):
        return None
    expected_samples_sha256 = dataset_receipt.get("samples_sha256")
    manifest = dataset_receipt.get("manifest") or {}
    if expected_samples_sha256 is None:
        expected_samples_sha256 = (
            (manifest.get("artifacts") or {}).get("samples.tsv") or {}
        ).get("sha256")
    if not expected_samples_sha256 or audit.get("samples_sha256") != expected_samples_sha256:
        return None
    expected_rows = manifest.get("rows")
    if expected_rows is not None and int(audit.get("rows", -1)) != int(expected_rows):
        return None
    reused = dict(audit)
    reused["reused_existing_task_audit"] = True
    return reused


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sequence_signature(row):
    first = str(row.get("sequence_sha256") or "").strip()
    second = str(row.get("sequence_b_sha256") or "").strip()
    if not first:
        return None
    return f"{first}:{second}" if second else first


def audit_task_dataset(task, final_root):
    final_root = Path(final_root).resolve(); task_id = task["task_id"]
    samples = final_root / "datasets" / task_id / "samples.tsv"
    dataset_receipt = final_root / "dataset_receipts" / f"{task_id}.json"
    if not samples.is_file() or not dataset_receipt.is_file():
        return {"task_id": task_id, "status": "waiting", "reason": "dataset_not_ready"}
    work = final_root / "dataset_audit_work"; work.mkdir(parents=True, exist_ok=True)
    database = work / f"{task_id}.sqlite"
    if database.exists(): database.unlink()
    connection = sqlite3.connect(str(database))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE samples(sample_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
    connection.execute("CREATE TABLE signatures(signature TEXT NOT NULL, split TEXT NOT NULL)")
    connection.execute("CREATE TABLE groups_table(group_id TEXT NOT NULL, split TEXT NOT NULL)")
    connection.execute("CREATE INDEX signature_index ON signatures(signature)")
    connection.execute("CREATE INDEX group_index ON groups_table(group_id)")
    split_counts = Counter(); missing_sample_ids = 0; missing_signatures = 0
    duplicate_sample_ids = 0; rows = 0; batch_samples = []; batch_signatures = []; batch_groups = []
    try:
        with samples.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"sample_id", "split"}
            if required - set(reader.fieldnames or []):
                raise RuntimeError(f"{task_id} samples.tsv lacks {sorted(required)}")
            for row in reader:
                rows += 1
                sample_id = str(row.get("sample_id") or "").strip()
                split = _SPLIT_ALIASES.get(str(row.get("split") or "").strip(), str(row.get("split") or "").strip())
                if split not in {"train", "validation", "test"}:
                    raise RuntimeError(f"{task_id} invalid split {split!r}")
                split_counts[split] += 1
                if not sample_id:
                    missing_sample_ids += 1
                else:
                    batch_samples.append((sample_id, split))
                signature = _sequence_signature(row)
                if signature is None:
                    missing_signatures += 1
                else:
                    batch_signatures.append((signature, split))
                group_id = str(row.get("group_id") or row.get("assembly_id") or "").strip()
                if group_id:
                    batch_groups.append((group_id, split))
                if rows % 10000 == 0:
                    connection.executemany("INSERT OR IGNORE INTO samples VALUES (?,?)", batch_samples)
                    connection.executemany("INSERT INTO signatures VALUES (?,?)", batch_signatures)
                    connection.executemany("INSERT INTO groups_table VALUES (?,?)", batch_groups)
                    connection.commit(); batch_samples = []; batch_signatures = []; batch_groups = []
        connection.executemany("INSERT OR IGNORE INTO samples VALUES (?,?)", batch_samples)
        connection.executemany("INSERT INTO signatures VALUES (?,?)", batch_signatures)
        connection.executemany("INSERT INTO groups_table VALUES (?,?)", batch_groups)
        connection.commit()
        exact_cross_split = connection.execute(
            "SELECT COUNT(*) FROM (SELECT signature FROM signatures GROUP BY signature HAVING COUNT(DISTINCT split)>1)"
        ).fetchone()[0]
        group_cross_split = connection.execute(
            "SELECT COUNT(*) FROM (SELECT group_id FROM groups_table GROUP BY group_id HAVING COUNT(DISTINCT split)>1)"
        ).fetchone()[0]
        unique_samples = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        duplicate_sample_ids = rows - missing_sample_ids - unique_samples
    finally:
        connection.close()
        if database.exists(): database.unlink()
    cv_policy = _cv_policy(task)
    failures = []
    if rows == 0: failures.append("empty_dataset")
    if missing_sample_ids: failures.append("missing_sample_ids")
    if duplicate_sample_ids or unique_samples != rows: failures.append("duplicate_sample_ids")
    if missing_signatures: failures.append("missing_sequence_signatures")
    if exact_cross_split: failures.append("exact_sequence_cross_split")
    if cv_policy == "fixed_train_validation_test" and any(split_counts[name] == 0 for name in ("train", "validation", "test")):
        failures.append("incomplete_fixed_split")
    if cv_policy == "official_test_only_zero_shot" and (
        split_counts["test"] == 0
        or split_counts["train"] != 0
        or split_counts["validation"] != 0
    ):
        failures.append("invalid_zero_shot_official_test_split")
    if task["task_kind"] == "candidate_ranking" and group_cross_split:
        failures.append("candidate_group_cross_split")
    payload = {
        "status": "ok" if not failures else "failed",
        "protocol_id": DATASET_AUDIT_PROTOCOL_ID, "task_id": task_id,
        "task_kind": task["task_kind"], "split_policy": task.get("split_policy"),
        "cv_policy": cv_policy, "rows": rows, "unique_sample_ids": unique_samples,
        "split_counts": dict(sorted(split_counts.items())),
        "missing_sample_ids": missing_sample_ids, "duplicate_sample_ids": duplicate_sample_ids,
        "missing_sequence_signatures": missing_signatures,
        "exact_sequence_cross_split": int(exact_cross_split),
        "group_cross_split": int(group_cross_split),
        "group_cross_split_is_informational": task["task_kind"] != "candidate_ranking",
        "samples_path": str(samples.resolve()), "samples_sha256": sha256_path(samples),
        "dataset_receipt_path": str(dataset_receipt.resolve()),
        "dataset_receipt_sha256": sha256_path(dataset_receipt),
        "failures": failures,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    payload["receipt_sha256"] = _canonical_sha(payload)
    path = final_root / "dataset_audits" / f"{task_id}.json"; _atomic_json(path, payload)
    return payload


def audit_all_datasets(registry, final_root):
    final_root = Path(final_root).resolve(); records = []; reused_task_audits = 0
    for task in registry["tasks"]:
        if task["task_id"] in EXCLUDED_TASK_IDS:
            records.append({"task_id": task["task_id"], "status": "excluded_edta"})
        elif task["task_id"] == "B17":
            records.append({"task_id": task["task_id"], "status": "analysis_only"})
        else:
            existing = _valid_existing_task_audit(task, final_root)
            if existing is not None:
                records.append(existing)
                reused_task_audits += 1
                continue
            try: records.append(audit_task_dataset(task, final_root))
            except Exception as error:
                records.append({"task_id": task["task_id"], "status": "failed", "reason": repr(error)})
    artifacts = []
    for record in records:
        path = final_root / "dataset_audits" / f"{record['task_id']}.json"
        if record["status"] in {"ok", "failed"} and path.is_file():
            artifacts.append({
                "task_id": record["task_id"], "path": str(path),
                "sha256": sha256_path(path), "size_bytes": path.stat().st_size,
            })
    status = "ok" if all(record["status"] in {"ok", "excluded_edta", "analysis_only"} for record in records) else "waiting"
    receipt = {
        "status": status, "protocol_id": DATASET_AUDIT_PROTOCOL_ID,
        "ready": sum(record["status"] == "ok" for record in records),
        "reused_task_audits": reused_task_audits,
        "waiting_or_failed": sum(record["status"] not in {"ok", "excluded_edta", "analysis_only"} for record in records),
        "records": records, "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    _atomic_json(final_root / "DATASET_INTEGRITY_AUDIT.json", receipt)
    return receipt


def valid_dataset_audit(registry, final_root, deep=False, expected_task_ids=None):
    final_root = Path(final_root).resolve(); path = final_root / "DATASET_INTEGRITY_AUDIT.json"
    try: receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError): return None
    canonical = dict(receipt); stored = canonical.pop("receipt_sha256", None)
    expected = (
        set(map(str, expected_task_ids))
        if expected_task_ids is not None
        else {
            task["task_id"] for task in registry["tasks"]
            if task["task_id"] not in EXCLUDED_TASK_IDS and task["task_id"] != "B17"
        }
    )
    observed = set()
    if (
        receipt.get("status") != "ok"
        or receipt.get("protocol_id") != DATASET_AUDIT_PROTOCOL_ID
        or stored != _canonical_sha(canonical)
    ):
        return None
    for artifact in receipt.get("artifacts") or []:
        audit_path = Path(artifact["path"]); observed.add(artifact["task_id"])
        if (
            not audit_path.is_file()
            or audit_path.stat().st_size != int(artifact["size_bytes"])
            or (deep and sha256_path(audit_path) != artifact["sha256"])
        ):
            return None
        task_receipt = json.loads(audit_path.read_text(encoding="utf-8"))
        task_canonical = dict(task_receipt); task_stored = task_canonical.pop("receipt_sha256", None)
        samples = Path(task_receipt.get("samples_path", "")); dataset_receipt = Path(task_receipt.get("dataset_receipt_path", ""))
        if (
            task_receipt.get("status") != "ok"
            or task_stored != _canonical_sha(task_canonical)
            or not samples.is_file()
            or (deep and sha256_path(samples) != task_receipt.get("samples_sha256"))
            or not dataset_receipt.is_file()
            or (deep and sha256_path(dataset_receipt) != task_receipt.get("dataset_receipt_sha256"))
        ):
            return None
    return receipt if expected.issubset(observed) else None
