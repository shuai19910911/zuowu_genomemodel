"""Leakage-aware low-homology sensitivity cohorts for B13--B16."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .data import read_canonical_rows, sha256_path


PARENT_TASKS = ("B13", "B14", "B15", "B16")
MIN_IDENTITY = 0.80
MIN_QUERY_COVERAGE = 0.80
LOW_HOMOLOGY_PROTOCOL_ID = "mmseqs_test_vs_trainval_80id_80qcov_v2"


def _canonical_sha(payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def classify_low_homology(sample_ids, hits, min_identity=MIN_IDENTITY,
                            min_query_coverage=MIN_QUERY_COVERAGE):
    return {
        sample_id: not (
            sample_id in hits
            and float(hits[sample_id]["fident"]) >= float(min_identity)
            and float(hits[sample_id]["qcov"]) >= float(min_query_coverage)
        )
        for sample_id in sample_ids
    }


def preserve_ranking_groups(rows, low_flags):
    groups = {}
    for row in rows:
        group_id = row.get("group_id")
        if group_id:
            groups.setdefault(group_id, []).append(row)
    if not groups:
        return low_flags, "sample_level"
    result = dict(low_flags)
    for members in groups.values():
        positives = [row for row in members if int(float(row["label"])) == 1]
        if not positives:
            raise RuntimeError("candidate-ranking group has no positive member")
        group_low = all(low_flags[row["sample_id"]] for row in positives)
        for row in members:
            result[row["sample_id"]] = group_low
    return result, "whole_candidate_group_selected_by_positive_sequence"


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _paired_sequence(row):
    sequence = row["sequence"]
    if row.get("sequence_b"):
        sequence += "N" * 50 + row["sequence_b"]
    return sequence


def _write_fasta(rows, path, id_prefix):
    mapping = []
    with Path(path).open("w", encoding="ascii") as handle:
        for index, row in enumerate(rows):
            sequence_id = f"{id_prefix}{index:08d}"
            handle.write(f">{sequence_id}\n{_paired_sequence(row)}\n")
            mapping.append((sequence_id, row["sample_id"]))
    return mapping


def build_task_cohort(task_id, dataset_root, output_root, mmseqs="mmseqs", threads=8):
    if task_id not in PARENT_TASKS:
        raise ValueError(f"low-homology cohort is restricted to {PARENT_TASKS}")
    dataset_root = Path(dataset_root).resolve(); output_root = Path(output_root).resolve()
    samples_path = dataset_root / "samples.tsv"
    rows = read_canonical_rows(samples_path)
    references = [row for row in rows if row["split"] in {"train", "validation"}]
    tests = [row for row in rows if row["split"] == "test"]
    if not references or not tests:
        raise RuntimeError("low-homology cohort requires reference and test rows")
    output_root.mkdir(parents=True, exist_ok=True)
    work = output_root / f"work_{task_id}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    reference_fasta = work / "reference.fa"; test_fasta = work / "test.fa"
    _write_fasta(references, reference_fasta, "r")
    test_mapping = _write_fasta(tests, test_fasta, "q")
    result_path = work / "search.tsv"; temporary = work / "mmseqs_tmp"
    command = [
        str(mmseqs), "easy-search", str(test_fasta), str(reference_fasta),
        str(result_path), str(temporary), "--search-type", "3", "--threads", str(int(threads)),
        "--min-seq-id", str(MIN_IDENTITY), "-c", str(MIN_QUERY_COVERAGE),
        "--cov-mode", "0", "-s", "7.5",
        "--max-seqs", "1", "--format-output", "query,target,fident,qcov,tcov,evalue,bits",
    ]
    subprocess.run(command, check=True, cwd=work)
    query_to_sample = dict(test_mapping)
    hits = {}
    if result_path.is_file():
        with result_path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 7:
                    raise RuntimeError(f"invalid MMseqs row: {line!r}")
                sample_id = query_to_sample[fields[0]]
                candidate = {
                    "target": fields[1], "fident": float(fields[2]) / 100.0,
                    "qcov": float(fields[3]), "tcov": float(fields[4]),
                    "evalue": float(fields[5]), "bits": float(fields[6]),
                }
                if sample_id not in hits or candidate["bits"] > hits[sample_id]["bits"]:
                    hits[sample_id] = candidate
    sample_ids = [row["sample_id"] for row in tests]
    low_flags = classify_low_homology(sample_ids, hits)
    if task_id in {"B14", "B15"}:
        low_flags, cohort_unit = preserve_ranking_groups(tests, low_flags)
    else:
        cohort_unit = "sample_level"
    cohort_path = output_root / f"{task_id}.low_homology.tsv"
    temporary_cohort = cohort_path.with_suffix(cohort_path.suffix + ".tmp")
    fields = ["task_id", "sample_id", "low_homology", "best_identity", "best_query_coverage", "best_target"]
    with temporary_cohort.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for sample_id in sample_ids:
            hit = hits.get(sample_id, {})
            writer.writerow({
                "task_id": task_id, "sample_id": sample_id,
                "low_homology": int(low_flags[sample_id]),
                "best_identity": hit.get("fident", ""),
                "best_query_coverage": hit.get("qcov", ""),
                "best_target": hit.get("target", ""),
            })
    os.replace(temporary_cohort, cohort_path)
    retained_hits = output_root / f"{task_id}.mmseqs_hits.tsv"
    shutil.copyfile(result_path, retained_hits)
    version = subprocess.run([str(mmseqs), "version"], check=True, text=True, capture_output=True).stdout.strip()
    receipt = {
        "status": "ok", "task_id": task_id, "method": "MMseqs2 nucleotide easy-search",
        "protocol_id": LOW_HOMOLOGY_PROTOCOL_ID,
        "reference_splits": ["train", "validation"], "test_samples": len(tests),
        "reference_samples": len(references), "low_homology_samples": sum(low_flags.values()),
        "min_identity": MIN_IDENTITY, "min_query_coverage": MIN_QUERY_COVERAGE,
        "cohort_unit": cohort_unit,
        "mmseqs_version": version, "command": command,
        "samples_path": str(samples_path), "samples_sha256": sha256_path(samples_path),
        "cohort_path": str(cohort_path), "cohort_sha256": sha256_path(cohort_path),
        "hits_path": str(retained_hits), "hits_sha256": sha256_path(retained_hits),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    receipt_path = output_root / f"{task_id}.RECEIPT.json"
    _atomic_json(receipt_path, receipt)
    shutil.rmtree(work)
    return receipt
