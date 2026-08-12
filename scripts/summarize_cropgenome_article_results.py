#!/usr/bin/env python3
"""Create category-aware article-guided summary after the matrix completes."""

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_server_transfer"))

from downstream_v4.article_analysis import build_article_summary
from downstream_v4.article_protocol import ARTICLE_GUIDED_PROTOCOL_ID


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-root",
        default=str(ROOT / "training_server_transfer/runs/cropgenome_downstream_final_v1"),
    )
    args = parser.parse_args()
    final_root = Path(args.final_root).resolve()
    plan = json.loads((final_root / "FINAL_PROTOCOL.json").read_text(encoding="utf-8"))
    if plan.get("protocol_id") != ARTICLE_GUIDED_PROTOCOL_ID:
        raise RuntimeError("FINAL_PROTOCOL is not article-guided")
    receipts = []
    for row in plan.get("rows") or []:
        path = final_root / "results" / row["run_key"] / "FINAL_RECEIPT.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        metrics = payload.get("test_metrics") or {}
        if not metrics:
            try:
                nested = json.loads(
                    Path(payload["result_receipt"]).read_text(encoding="utf-8")
                )
            except (OSError, ValueError, KeyError):
                nested = {}
            metrics = nested.get("test_metrics") or {}
        receipts.append({"run_key": row["run_key"], "test_metrics": metrics})
    summary = build_article_summary(plan, receipts)
    output = final_root / "controller/ARTICLE_GUIDED_SUMMARY.json"
    _write_json(output, summary)
    print(json.dumps({
        "status": summary["status"],
        "output": str(output),
        "observed": summary["observed_classification_rows"],
        "expected": summary["expected_classification_rows"],
        "final_leaderboard_allowed": summary["final_leaderboard_allowed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
