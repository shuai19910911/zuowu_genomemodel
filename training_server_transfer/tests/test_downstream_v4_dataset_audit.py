import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.dataset_audit import (
    audit_all_datasets, audit_task_dataset, valid_dataset_audit,
)


def _dataset(root, rows, task_id="T"):
    dataset = root / "datasets" / task_id; dataset.mkdir(parents=True)
    fields = ["sample_id", "split", "sequence_sha256", "group_id"]
    with (dataset / "samples.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    receipts = root / "dataset_receipts"; receipts.mkdir()
    (receipts / f"{task_id}.json").write_text(json.dumps({"status": "ok", "task_id": task_id}))


def _task(kind="binary_classification"):
    return {"task_id": "T", "task_kind": kind, "split_policy": "official"}


def test_dataset_audit_accepts_unique_hash_separated_fixed_split(tmp_path):
    _dataset(tmp_path, [
        {"sample_id": "a", "split": "train", "sequence_sha256": "1", "group_id": "g1"},
        {"sample_id": "b", "split": "val", "sequence_sha256": "2", "group_id": "g1"},
        {"sample_id": "c", "split": "test", "sequence_sha256": "3", "group_id": "g1"},
    ])
    receipt = audit_task_dataset(_task(), tmp_path)
    assert receipt["status"] == "ok"
    assert receipt["group_cross_split"] == 1
    assert receipt["group_cross_split_is_informational"] is True
    overall = audit_all_datasets({"tasks": [_task()]}, tmp_path)
    assert overall["status"] == "ok"
    assert valid_dataset_audit({"tasks": [_task()]}, tmp_path)


def test_dataset_audit_rejects_exact_sequence_or_ranking_group_leakage(tmp_path):
    _dataset(tmp_path, [
        {"sample_id": "a", "split": "train", "sequence_sha256": "same", "group_id": "g"},
        {"sample_id": "b", "split": "validation", "sequence_sha256": "2", "group_id": "g"},
        {"sample_id": "c", "split": "test", "sequence_sha256": "same", "group_id": "g"},
    ])
    receipt = audit_task_dataset(_task("candidate_ranking"), tmp_path)
    assert receipt["status"] == "failed"
    assert "exact_sequence_cross_split" in receipt["failures"]
    assert "candidate_group_cross_split" in receipt["failures"]


def test_dataset_audit_rejects_duplicate_sample_ids(tmp_path):
    _dataset(tmp_path, [
        {"sample_id": "a", "split": "train", "sequence_sha256": "1", "group_id": ""},
        {"sample_id": "a", "split": "validation", "sequence_sha256": "2", "group_id": ""},
        {"sample_id": "c", "split": "test", "sequence_sha256": "3", "group_id": ""},
    ])
    receipt = audit_task_dataset(_task(), tmp_path)
    assert receipt["status"] == "failed"
    assert receipt["duplicate_sample_ids"] == 1


def test_dataset_audit_accepts_official_test_only_zero_shot_panel(tmp_path):
    _dataset(tmp_path, [
        {"sample_id": "a", "split": "test", "sequence_sha256": "1", "group_id": "g1"},
        {"sample_id": "b", "split": "test", "sequence_sha256": "2", "group_id": "g2"},
    ])
    task = _task("zero_shot_variant")
    receipt = audit_task_dataset(task, tmp_path)
    assert receipt["status"] == "ok"
    assert receipt["cv_policy"] == "official_test_only_zero_shot"
    assert receipt["split_counts"] == {"test": 2}


def test_dataset_audit_rejects_train_rows_in_official_zero_shot_panel(tmp_path):
    _dataset(tmp_path, [
        {"sample_id": "a", "split": "train", "sequence_sha256": "1", "group_id": "g1"},
        {"sample_id": "b", "split": "test", "sequence_sha256": "2", "group_id": "g2"},
    ])
    receipt = audit_task_dataset(_task("zero_shot_binary"), tmp_path)
    assert receipt["status"] == "failed"
    assert "invalid_zero_shot_official_test_split" in receipt["failures"]
