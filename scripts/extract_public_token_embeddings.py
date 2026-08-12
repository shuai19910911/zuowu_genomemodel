#!/usr/bin/env python3
"""Extract aligned per-base hidden states from compatible public DNA models for B13."""

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from extract_public_dna_embeddings import (
    canonicalize_rcps_hidden, load_hf_model, model_hidden, public_model_snapshot_sha256,
    sha256_path, supported_forward_kwargs, verified_sha256_path,
)
from training_server_transfer.downstream_v4.model_adapters import clean_model_runtime_cache


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def extract_public_token_cache(model, tokenizer, samples, output_dir, context, batch_size,
                               device, model_id, inference_dtype):
    shared = _module(ROOT / "scripts/extract_cropgenome_structure_token_embeddings.py", "crop_token_shared_public")
    output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    context = int(context); hidden_path = output_dir / "hidden.f16"; labels_path = output_dir / "labels.u1"
    labels_cache = np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(len(samples), context))
    hidden_cache = None; hidden_dim = None; started = time.time()
    bases = np.asarray(list("ACGTN"))
    for start in range(0, len(samples), int(batch_size)):
        batch_rows = samples[start:start + int(batch_size)]
        tokens = [shared._read_slice(row, "input_path", "offset", "length", np.uint8, context) for row in batch_rows]
        labels = [shared._read_slice(row, "label_path", "label_offset", "label_length", np.uint8, context) for row in batch_rows]
        sequences = ["".join(bases[array].tolist()) for array in tokens]
        encoded = tokenizer(
            sequences, add_special_tokens=False, padding=False, truncation=False,
            return_attention_mask=True, return_tensors="pt",
        )
        if encoded["input_ids"].shape != (len(batch_rows), context):
            raise RuntimeError(
                f"not_applicable: tokenizer is not one-token-per-base; got {tuple(encoded['input_ids'].shape)}, "
                f"expected {(len(batch_rows), context)}"
            )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        kwargs = supported_forward_kwargs(model, {**encoded, "output_hidden_states": False, "return_dict": True})
        autocast_enabled = device.type == "cuda" and inference_dtype in {
            torch.float16, torch.bfloat16,
        }
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=inference_dtype if autocast_enabled else torch.float16,
            enabled=autocast_enabled,
        ):
            outputs = model(**kwargs)
            hidden = model_hidden(outputs)
            hidden = canonicalize_rcps_hidden(
                hidden, d_model=getattr(model.config, "d_model", None),
                rcps=bool(getattr(model.config, "rcps", False)),
            )
        hidden_np = hidden.float().cpu().numpy()
        if hidden_np.shape[:2] != (len(batch_rows), context) or not np.isfinite(hidden_np).all():
            raise RuntimeError(f"invalid aligned hidden tensor: {hidden_np.shape}")
        if hidden_cache is None:
            hidden_dim = int(hidden_np.shape[-1])
            hidden_cache = np.memmap(hidden_path, dtype=np.float16, mode="w+", shape=(len(samples), context, hidden_dim))
        stop = start + len(batch_rows)
        hidden_cache[start:stop] = hidden_np.astype(np.float16)
        labels_cache[start:stop] = np.stack(labels)
    hidden_cache.flush(); labels_cache.flush()
    sample_path = output_dir / "selected_samples.tsv"
    import csv
    fields = ["sample_id", "split", "assembly_id", "species", "genus", "target_label", "feature_id", "contig_id", "center0", "strand"]
    present = {key for row in samples for key in row}; fields = [key for key in fields if key in present]
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(samples)
    return {
        "status": "ok", "task_id": "B13", "cache_type": "aligned_token_hidden_memmap",
        "model_id": model_id, "samples": len(samples), "context": context,
        "hidden_shape": [len(samples), context, hidden_dim], "labels_shape": [len(samples), context],
        "hidden_dtype": "float16", "labels_dtype": "uint8", "finite_hidden": True,
        "hidden_path": str(hidden_path), "labels_path": str(labels_path),
        "selected_samples_path": str(sample_path), "hidden_sha256": sha256_path(hidden_path),
        "labels_sha256": sha256_path(labels_path), "selected_samples_sha256": sha256_path(sample_path),
        "elapsed_seconds": time.time() - started, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True); parser.add_argument("--hf-model", required=True)
    parser.add_argument("--weight-file", required=True); parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--weight-verification-cache")
    parser.add_argument("--model-snapshot-sha256", required=True); parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--max-per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260730); parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-head", choices=["auto", "masked-lm-base"], default="auto")
    parser.add_argument("--load-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--inference-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--disable-remote-triton", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true"); parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    model_root = Path(args.hf_model).resolve(); weight = model_root / args.weight_file
    clean_model_runtime_cache(model_root)
    try:
        verified_sha256_path(weight, args.weight_sha256, args.weight_verification_cache)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    if public_model_snapshot_sha256(model_root, args.weight_file) != args.model_snapshot_sha256:
        raise SystemExit("public model snapshot SHA256 mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    dtype_by_name = {
        "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
    }
    dtype = dtype_by_name[args.inference_dtype] if device.type == "cuda" else torch.float32
    load_dtype = dtype_by_name[args.load_dtype]
    if device.type != "cuda": load_dtype = torch.float32
    tokenizer, model, compatibility = load_hf_model(
        str(model_root), None, args.trust_remote_code, load_dtype, args.local_files_only, args.model_head,
    )
    if args.disable_remote_triton:
        import importlib
        model_module = importlib.import_module(model.__class__.__module__)
        if not hasattr(model_module, "flash_attn_qkvpacked_func"):
            raise SystemExit("model module has no flash_attn_qkvpacked_func fallback switch")
        model_module.flash_attn_qkvpacked_func = None
    model = model.eval().to(device=device, dtype=dtype)
    shared = _module(ROOT / "scripts/extract_cropgenome_structure_token_embeddings.py", "crop_token_selection_public")
    selected = shared.select_segmentation_samples(
        shared.read_samples(args.samples), max_per_group=args.max_per_group, seed=args.seed,
    )
    manifest = extract_public_token_cache(
        model, tokenizer, selected, args.output_dir, args.context, args.batch_size,
        device, args.model_id, dtype,
    )
    manifest.update({
        "model_root": str(model_root), "weight_file": args.weight_file,
        "weight_sha256": args.weight_sha256, "model_snapshot_sha256": args.model_snapshot_sha256,
        "load_compatibility_mode": compatibility, "selection_seed": args.seed,
        "selection_max_per_group": args.max_per_group, "device": str(device),
        "implementation_sha256": sha256_path(Path(__file__).resolve()),
    })
    (Path(args.output_dir) / "cache_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
