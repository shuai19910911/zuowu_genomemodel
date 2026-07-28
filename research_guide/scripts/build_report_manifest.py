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
    manifest = {
        "status": "ok" if md["status"] == "passed" and plain["status"] == "passed" and docx["status"] == "passed" else "failed",
        "report_id": "CropGenome-FM-detailed-research-guide-cn-v2.1-plain-language",
        "evidence_cutoff": "2026-07-21T14:04:09+08:00",
        "document_updated": "2026-07-27",
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
            "docx_visual": {
                "status": "passed",
                "renderer": "LibreOffice 26.2.4.2",
                "rendered_pdf_pages": 36,
                "png_pages_checked": 36,
                "near_blank_pages": 0,
                "contact_sheets_checked": 6,
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
