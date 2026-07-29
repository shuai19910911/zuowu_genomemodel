#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
import struct
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "research_guide"
MARKDOWN = GUIDE / "README_CN.md"
DOCX = GUIDE / "CropGenome-FM_详细研究设计与评估报告_CN.docx"
OUTPUT = GUIDE / "report_manifest.json"
SHARD_AUDIT = GUIDE / "source_data" / "stage_b_shard_sampling_audit.tsv"
REGION_COVERAGE = GUIDE / "source_data" / "stage_b_region_training_coverage.tsv"
SPLIT_COVERAGE = GUIDE / "source_data" / "stage_b_split_assembly_coverage.tsv"
ASSEMBLY_SUMMARY = GUIDE / "source_data" / "assembly_species_summary.tsv"
TRAIN_SCRIPT = ROOT / "training_server_transfer" / "scripts" / "train.py"
WINDOW_BUILDER = ROOT / "scripts" / "build_stage_window_candidates.py"
REGION_BUILDER = ROOT / "scripts" / "build_region_candidates.py"
STAGE_ENCODER = ROOT / "scripts" / "encode_stage_inputs.py"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()[:24]
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not PNG: {path}")
    return struct.unpack(">II", raw[16:24])


def count_tsv(path: Path) -> tuple[int, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    return max(0, len(rows) - 1), len(rows[0]) if rows else 0


def stage_b_shard_sampling_stats() -> dict[str, object]:
    with SHARD_AUDIT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    with REGION_COVERAGE.open(encoding="utf-8", newline="") as handle:
        region_rows = list(csv.DictReader(handle, delimiter="\t"))
    with SPLIT_COVERAGE.open(encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle, delimiter="\t"))
    with ASSEMBLY_SUMMARY.open(encoding="utf-8", newline="") as handle:
        assembly_rows = list(csv.DictReader(handle, delimiter="\t"))
    train_code = TRAIN_SCRIPT.read_text(encoding="utf-8")
    sampler_snippets = [
        'row["windows_count"]',
        "weights / weights.sum()",
        "load_window_index(windows_path, self.split",
        "rng.choice(len(self.shards), p=self.shard_probs)",
        "rng.integers(0, len(offsets))",
    ]
    tokens_total = sum(int(row["tokens"]) for row in rows)
    windows_total = sum(int(row["windows_total"]) for row in rows)
    train_windows = sum(int(row["train_windows"]) for row in rows)
    val_windows = sum(int(row["val_windows"]) for row in rows)
    test_windows = sum(int(row["test_windows"]) for row in rows)
    full_rows = [row for row in rows if int(row["tokens"]) >= 900_000_000]
    weights = [float(row["relative_train_sampling_weight"]) for row in rows]
    region_by_name = {row["region"]: row for row in region_rows}
    split_by_name = {row["split"]: row for row in split_rows}
    expected_region_all = {
        "background": 223_487, "coding": 1_790_629, "gene_body": 559_494,
        "promoter": 783_111, "splice": 1_175_065, "tes": 447_604, "utr": 615_391,
    }
    expected_region_train = {
        "background": 219_169, "coding": 1_755_568, "gene_body": 548_536,
        "promoter": 767_774, "splice": 1_152_040, "tes": 438_839, "utr": 603_314,
    }
    expected_step14000 = {
        "background": 20_040.432, "coding": 161_083.532, "gene_body": 50_314.052,
        "promoter": 70_416.248, "splice": 105_693.289, "tes": 40_246.474, "utr": 55_205.972,
    }
    expected_step17000 = {
        "background": 24_343.348, "coding": 195_670.056, "gene_body": 61_117.069,
        "promoter": 85_535.442, "splice": 128_386.878, "tes": 48_887.864, "utr": 67_059.342,
    }
    expected_splits = {
        "train": (180, 5_485_240), "val": (29, 54_695), "test": (29, 54_846),
    }
    planned_assemblies = {
        "total": sum(int(row["assemblies"]) for row in assembly_rows),
        "train": sum(int(row["train"]) for row in assembly_rows),
        "validation": sum(int(row["validation"]) for row in assembly_rows),
        "test": sum(int(row["test"]) for row in assembly_rows),
    }
    checks = {
        "rows_42": len(rows) == 42,
        "full_shards_41": len(full_rows) == 41,
        "tokens_total_exact": tokens_total == 41_242_505_216,
        "windows_total_exact": windows_total == 5_594_781,
        "train_windows_exact": train_windows == 5_485_240,
        "val_windows_exact": val_windows == 54_695,
        "test_windows_exact": test_windows == 54_846,
        "full_shard_token_min_exact": min(int(row["tokens"]) for row in full_rows) == 999_993_344,
        "full_shard_token_max_exact": max(int(row["tokens"]) for row in full_rows) == 999_997_440,
        "full_shard_window_min_exact": min(int(row["windows_total"]) for row in full_rows) == 122_062,
        "full_shard_window_max_exact": max(int(row["windows_total"]) for row in full_rows) == 141_012,
        "relative_weight_min_matches": abs(min(weights) - 0.980420860) < 1e-9,
        "relative_weight_max_matches": abs(max(weights) - 1.067065238) < 1e-9,
        "sampler_code_shape_present": all(snippet in train_code for snippet in sampler_snippets),
        "seven_region_rows": set(region_by_name) == set(expected_region_all),
        "region_all_counts_exact": all(
            int(region_by_name[name]["all_pool_windows"]) == value
            for name, value in expected_region_all.items()
        ),
        "region_train_counts_exact": all(
            int(region_by_name[name]["train_pool_windows"]) == value
            for name, value in expected_region_train.items()
        ),
        "step14000_expected_draws_exact": all(
            abs(float(region_by_name[name]["expected_draws_step14000"]) - value) < 0.0011
            for name, value in expected_step14000.items()
        ),
        "step17000_expected_draws_exact": all(
            abs(float(region_by_name[name]["expected_draws_step17000"]) - value) < 0.0011
            for name, value in expected_step17000.items()
        ),
        "region_draws_explicitly_not_observed": all(
            row["actual_draw_count_recorded"] == "no"
            and row["evidence_boundary"] == "expected_from_verified_sampler_not_an_observed_draw_log"
            for row in region_rows
        ),
        "split_rows_exact": set(split_by_name) == set(expected_splits),
        "split_assembly_and_window_counts_exact": all(
            int(split_by_name[name]["assemblies_with_windows"]) == expected[0]
            and int(split_by_name[name]["windows"]) == expected[1]
            for name, expected in expected_splits.items()
        ),
        "split_assembly_overlap_zero": all(
            int(row["assembly_overlap_with_other_splits"]) == 0 for row in split_rows
        ),
        "planned_assembly_split_exact": planned_assemblies == {
            "total": 258, "train": 192, "validation": 35, "test": 31,
        },
        "actual_contributing_assemblies_238": sum(
            int(row["assemblies_with_windows"]) for row in split_rows
        ) == 238,
        "data_builder_hashes_exact": {
            "window_builder": sha256(WINDOW_BUILDER),
            "region_builder": sha256(REGION_BUILDER),
            "stage_encoder": sha256(STAGE_ENCODER),
        } == {
            "window_builder": "fe0b59af3c8176533cc42d193225851c3cee78f5d45e1db8de9dc146956de320",
            "region_builder": "64d92668247c5c75cd805afdabb2892331ecd4bddbf8200d96d90519b8048a36",
            "stage_encoder": "100e1aacaf3ba81ab921a6d24f1e2ce7f493a43282c1b738d1325af9159d647e",
        },
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "rows": len(rows),
        "tokens_total": tokens_total,
        "windows_total": windows_total,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "test_windows": test_windows,
        "full_shard_tokens_range": [
            min(int(row["tokens"]) for row in full_rows),
            max(int(row["tokens"]) for row in full_rows),
        ],
        "full_shard_windows_range": [
            min(int(row["windows_total"]) for row in full_rows),
            max(int(row["windows_total"]) for row in full_rows),
        ],
        "relative_train_sampling_weight_range": [min(weights), max(weights)],
        "audit_sha256": sha256(SHARD_AUDIT),
        "region_coverage_sha256": sha256(REGION_COVERAGE),
        "split_assembly_coverage_sha256": sha256(SPLIT_COVERAGE),
        "planned_assembly_split": planned_assemblies,
        "actual_contributing_assembly_split": {
            name: int(row["assemblies_with_windows"]) for name, row in split_by_name.items()
        },
        "region_training_coverage_rows": len(region_rows),
        "sampler_implementation_sha256": sha256(TRAIN_SCRIPT),
        "data_builder_sha256": {
            "window_builder": sha256(WINDOW_BUILDER),
            "region_builder": sha256(REGION_BUILDER),
            "stage_encoder": sha256(STAGE_ENCODER),
        },
    }


def markdown_stats() -> dict[str, object]:
    text = MARKDOWN.read_text(encoding="utf-8")
    lines = text.splitlines()
    links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text) + re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    broken = []
    for target in links:
        if re.match(r"https?://", target) or target.startswith("#"):
            continue
        if not (GUIDE / target).exists():
            broken.append(target)
    return {
        "status": "passed" if not broken else "failed",
        "lines": len(lines),
        "characters": len(text),
        "tables": sum(1 for line in lines if re.match(r"^\|\s*:?-{3,}", line)),
        "images": len(re.findall(r"!\[[^\]]*\]\([^)]+\)", text)),
        "broken_relative_links": len(broken),
        "broken_targets": broken,
        "absolute_home_paths": text.count(str(Path.home()) + "/"),
    }


def plain_language_stats() -> dict[str, object]:
    text = MARKDOWN.read_text(encoding="utf-8")
    required = [
        "一眼看懂：我们具体是怎么做的",
        "DNA片段（window）",
        "数据分片（shard）",
        "它只是文件包装方式，没有额外生物学含义",
        "理论上可以写成一个约41.24 GB的大文件",
        "每个分片按碱基总数装满，不是按片段条数装满",
        "最低约为理想均匀值的0.980倍",
        "最高约为1.067倍",
        "原始258个版本的预设划分",
        "最终实际贡献Stage B片段",
        "当前构建流程没有做全数据集序列相似性去重",
        "没有按区域强制配额",
        "没有保存逐区域实际抽样计数",
        "预计到第14,000次抽取",
        "模型存档（checkpoint）",
        "简单预测头（probe）",
        "3个测试基因组版本都未用于预训练",
        "未来候选任务，不是已经完成的实验",
    ]
    banned_patterns = {
        "unexplained_shard_workflow": r"先按\s*shard|按shard窗口数|shard窗口数",
        "opaque_machine_status": (
            r"\b(?:current source available|task release not built|not frozen|"
            r"no current formal result|blocked until|currently blocked|planned_not_evaluated)\b"
        ),
        "raw_pipeline_jargon": (
            r"\b(?:IterableDataset|DataLoader|logits|hidden state|proxy label|zero-truncation)\b"
        ),
    }
    banned_hits = {
        name: [i + 1 for i, line in enumerate(text.splitlines()) if re.search(pattern, line, re.I)]
        for name, pattern in banned_patterns.items()
    }
    missing = [phrase for phrase in required if phrase not in text]
    return {
        "status": "passed" if not missing and not any(banned_hits.values()) else "failed",
        "required_plain_language_phrases_present": not missing,
        "missing_required_phrases": missing,
        "banned_phrase_line_numbers": banned_hits,
        "shard_mentions": len(re.findall(r"\bshard\b", text, re.I)),
    }


def docx_stats() -> dict[str, object]:
    crc_error = None
    with ZipFile(DOCX) as archive:
        bad = archive.testzip()
        if bad:
            crc_error = bad
        document = ET.fromstring(archive.read("word/document.xml"))
        text = "".join(node.text or "" for node in document.iter(f"{{{NS['w']}}}t"))
        counts = {
            "ooxml_paragraph_elements": len(document.findall(".//w:p", NS)),
            "tables": len(document.findall(".//w:tbl", NS)),
            "section_properties": len(document.findall(".//w:sectPr", NS)),
            "drawing_elements": len(document.findall(".//w:drawing", NS)),
        }
    required = [
        "369,505,287", "CropGenomeFM_step14000", "NT-v2 500M",
        "下游科、属、种独立性", "CROP-LONGGENE-SEG", "AgroNT 1B",
        "PlantCAD2", "Evo2 1B", "一眼看懂：我们具体是怎么做的",
        "数据分片（shard）", "3个测试基因组版本都未用于预训练",
        "理论上可以写成一个约41.24 GB的大文件",
        "每个分片按碱基总数装满，不是按片段条数装满",
        "最低约为理想均匀值的0.980倍", "最高约为1.067倍",
        "原始258个版本的预设划分", "最终实际贡献Stage B片段",
        "当前构建流程没有做全数据集序列相似性去重",
        "没有按区域强制配额", "没有保存逐区域实际抽样计数",
        "预计到第14,000次抽取",
    ]
    return {
        "status": "passed" if crc_error is None and all(x in text for x in required) else "failed",
        "counts": counts,
        "required_key_strings_present": all(x in text for x in required),
        "missing_key_strings": [x for x in required if x not in text],
        "absolute_home_paths": text.count(str(Path.home()) + "/"),
        "zip_crc_error": crc_error,
    }


def file_entry(path: Path) -> dict[str, object]:
    rel = path.relative_to(ROOT).as_posix()
    entry: dict[str, object] = {
        "path": rel,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if path == MARKDOWN:
        md = markdown_stats()
        entry.update({
            "lines": md["lines"],
            "characters": md["characters"],
            "markdown_tables": md["tables"],
            "markdown_images": md["images"],
        })
    elif path.suffix == ".tsv":
        rows, columns = count_tsv(path)
        entry.update({"rows": rows, "columns": columns})
    elif path.suffix == ".png":
        width, height = png_size(path)
        entry.update({"width_px": width, "height_px": height})
    return entry


def main() -> None:
    files = [
        p for p in GUIDE.rglob("*")
        if p.is_file() and p != OUTPUT and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
    ]
    md = markdown_stats()
    plain = plain_language_stats()
    docx = docx_stats()
    shard_sampling = stage_b_shard_sampling_stats()
    manifest = {
        "status": "ok" if md["status"] == "passed" and plain["status"] == "passed" and docx["status"] == "passed" and shard_sampling["status"] == "passed" else "failed",
        "report_id": "CropGenome-FM-detailed-research-guide-cn-v2.3-stage-b-data-and-region-coverage",
        "evidence_cutoff": "2026-07-21T14:04:09+08:00",
        "document_updated": "2026-07-29",
        "formal_test_rerun_for_document_update": False,
        "scope": {
            "completed_evidence": [
                "Stage B 8K selected checkpoint step14000",
                "CropGenome-Bench-v1 3 tasks x 3 nested contexts",
                "Plant Genomic Benchmark 7 tasks at capability-eligible contexts",
                "publication-v2 final verification and final manifest status ok",
            ],
            "new_v2_audits": [
                "10-task downstream taxonomy and Stage B overlap audit",
                "7-panel gap to current strongest plant baseline",
                "9-task baseline capability and preregistered success matrix",
            ],
            "new_v2_1_plain_language_changes": [
                "10-step plain-language end-to-end workflow",
                "Chinese-first definitions for shard/window/token/checkpoint/probe and related terms",
                "plain-language rewrites of sampling, architecture, training, evaluation, future tasks and stopping rules",
                "no metric, model or formal-test result changed",
            ],
            "new_v2_2_shard_sampling_corrections": [
                "corrected the claim that Stage B windows could not fit in one file",
                "documented the one-billion-base shard cap and variable window counts",
                "audited all 42 shard indexes and the historical split-filtered sampler",
                "disclosed the 0.980-1.067 relative train-window sampling weight range",
                "confirmed validation and test windows were filtered before parameter updates",
                "no metric, checkpoint or formal-test result changed",
            ],
            "new_v2_3_stage_b_data_and_region_coverage": [
                "corrected the 258-assembly planned split to 192 train, 35 validation and 31 test candidates",
                "audited 180, 29 and 29 assemblies actually contributing Stage B windows with zero cross-split assembly overlap",
                "documented annotation-derived regions, context selection, quotas, boundaries, sequence QC, encoding and the 30B-to-41.243B outcome",
                "published exact seven-region train-pool counts and expected draws at steps 14000 and 17000",
                "distinguished near-certain category exposure from absent historical per-region observed counters",
                "no metric, checkpoint or formal-test result changed",
            ],
            "partial_or_not_completed": [
                "Stage C1 64K stopped at step569 before first validation",
                "Stage C2 128K data only",
                "Stage D 256K data only",
                "NT-v2 500M planned but not evaluated",
                "9 future crop-specific task families designed but not run",
            ],
        },
        "source_bindings": {
            "assembly_manifest": "data_manifests/assemblies.canonical.tsv",
            "plant_genomic_benchmark_revision": "78ec8156c2ffb3e5475277fdb7eb603294224e53",
            "stage_b_manifest": "training_server_transfer/runs/Stage_B_cropgenome_fm_v2_stable/stage_B_8k_final_manifest.json",
            "stage_b_shard_sampling_audit": "research_guide/source_data/stage_b_shard_sampling_audit.tsv",
            "stage_b_region_training_coverage": "research_guide/source_data/stage_b_region_training_coverage.tsv",
            "stage_b_split_assembly_coverage": "research_guide/source_data/stage_b_split_assembly_coverage.tsv",
            "stage_b_window_candidate_builder_local_sha256": "fe0b59af3c8176533cc42d193225851c3cee78f5d45e1db8de9dc146956de320",
            "stage_b_region_candidate_builder_local_sha256": "64d92668247c5c75cd805afdabb2892331ecd4bddbf8200d96d90519b8048a36",
            "stage_b_encoder_local_sha256": "100e1aacaf3ba81ab921a6d24f1e2ce7f493a43282c1b738d1325af9159d647e",
            "stage_b_sampler_implementation": "training_server_transfer/scripts/train.py",
            "stage_b_local_input_manifest_sha256": "710c633b66c7ec8d3aa1465526cd32b6a97fc846eba964b2a0059c01dd68c2ff",
            "publication_v2_final_report": "PUBLICATION_V2_FINAL_RESULTS_CN.md",
            "core_full_metrics": "training_server_transfer/runs/cropgenome_bench_v1_8k_complete/summary/full_data_metrics.tsv",
            "external_full_metrics": "training_server_transfer/runs/plant_genomic_benchmark_publication_v2/summary/full_data_metrics.tsv",
            "v2_derived_table_builder": "research_guide/scripts/build_evidence_update_tables.py",
        },
        "validation": {
            "markdown": md,
            "plain_language": plain,
            "docx_schema": {
                "status": "passed",
                "validator": "docx skill OOXML validator",
                "validator_output": "All validations PASSED!",
                "counts": docx["counts"],
            },
            "docx_content": docx,
            "stage_b_shard_sampling": shard_sampling,
            "docx_visual": {
                "status": "passed",
                "renderer": "LibreOffice 26.2.4.2",
                "rendered_pdf_pages": 39,
                "png_pages_checked": 39,
                "near_blank_pages": 0,
                "contact_sheets_checked": 7,
                "numbered_lists_reset_per_markdown_section": True,
                "final_overflow_or_crop_errors": 0,
                "final_blank_pages": 0,
            },
            "figures": {
                "status": "passed",
                "png_pdf_pairs": 5,
                "visual_checks_completed": True,
                "strategy_figure_boundary_note": "planned context and data status are not observed performance",
            },
            "derived_tables": {
                "status": "passed",
                "taxonomy_detail_rows": 71,
                "taxonomy_summary_rows": 10,
                "baseline_gap_rows": 7,
                "future_task_capability_rows": 9,
                "stage_b_region_training_coverage_rows": 7,
                "stage_b_split_assembly_coverage_rows": 3,
                "crlf_rows": 0,
                "empty_cells": 0,
            },
        },
        "metric_integrity_note": {
            "issue": "legacy core formal AUPRC implementation is not tie-safe",
            "impact": "majority baseline and other tie-heavy baselines; learned-model ranking unchanged under tie-safe recomputation",
            "required_next_release_action": "replace with tested standard Average Precision and regenerate formal artifacts",
        },
        "files": [file_entry(path) for path in sorted(files)],
        "manifest_self_hash": "excluded_by_design",
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(files)} files)")


if __name__ == "__main__":
    main()
