#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path


STAGES = ["Stage_B", "Stage_C1", "Stage_C2", "Stage_D"]


def check_training_runtime():
    try:
        import numpy
        import torch
    except ImportError as exc:
        raise SystemExit(f"missing training dependency in {sys.executable}: {exc.name}") from exc
    return {
        "python": sys.executable,
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_manifest(stage_dir):
    rows = []
    with open(stage_dir / "manifest.tsv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rows.append(row)
    return rows


def check_stage(root, stage, quick):
    stage_dir = root / "inputs" / stage
    if not stage_dir.is_dir():
        raise SystemExit(f"missing stage dir: {stage_dir}")
    rows = read_manifest(stage_dir)
    if not rows:
        raise SystemExit(f"empty manifest: {stage_dir / 'manifest.tsv'}")
    total_tokens = 0
    total_windows = 0
    for row in rows:
        input_path = stage_dir / row["input_ids"]
        windows_path = stage_dir / row["windows"]
        if not input_path.exists():
            raise SystemExit(f"missing input shard: {input_path}")
        if not windows_path.exists():
            raise SystemExit(f"missing windows shard: {windows_path}")
        tokens = int(row["tokens"])
        if input_path.stat().st_size != tokens:
            raise SystemExit(f"token size mismatch: {input_path}")
        total_tokens += tokens
        total_windows += int(row["windows_count"])
        with gzip.open(windows_path, "rt", encoding="utf-8") as fh:
            header = fh.readline().strip().split("\t")
            required = {"stage", "split", "assembly_id", "contig_id", "offset", "length"}
            if not required.issubset(set(header)):
                raise SystemExit(f"bad windows header: {windows_path}")
        if not quick:
            if sha256(input_path) != row["input_sha256"]:
                raise SystemExit(f"input sha256 mismatch: {input_path}")
            if sha256(windows_path) != row["windows_sha256"]:
                raise SystemExit(f"windows sha256 mismatch: {windows_path}")
    summary = stage_dir / "summary.tsv"
    if not summary.exists():
        raise SystemExit(f"missing summary: {summary}")
    print(f"{stage}: OK shards={len(rows)} tokens={total_tokens} windows={total_windows}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--stage", choices=STAGES + ["all"], default="all")
    p.add_argument("--quick", action="store_true", help="skip full SHA256 of every shard")
    p.add_argument("--check-global-sha", action="store_true", help="run sha256sum -c SHA256SUMS")
    args = p.parse_args()
    root = Path(args.root).resolve()
    print(json.dumps({"training_runtime": check_training_runtime()}), flush=True)

    for rel in ["README.md", "MANIFEST.tsv", "SHA256SUMS", "metadata/token_vocab.tsv", "configs/model_large.json"]:
        path = root / rel
        if not path.exists():
            raise SystemExit(f"missing required file: {path}")
    model_cfg = load_json(root / "configs/model_large.json")
    if model_cfg["vocab"]["A"] != 0 or model_cfg["vocab"]["PAD"] != 6:
        raise SystemExit("unexpected token vocabulary in configs/model_large.json")
    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        cfg_name = "train_stage_" + stage.split("_", 1)[1] + ".json"
        cfg = load_json(root / "configs" / cfg_name)
        if cfg["stage"] != stage:
            raise SystemExit(f"stage mismatch in {cfg_name}")
        check_stage(root, stage, args.quick)
    if args.check_global_sha:
        subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=root, check=True)
    print("package_check: OK")


if __name__ == "__main__":
    main()
