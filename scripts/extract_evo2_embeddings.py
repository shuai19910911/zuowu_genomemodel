#!/usr/bin/env python3
"""Extract frozen Evo2 embeddings from a locally verified checkpoint.

The CentOS 7 host cannot load Transformer Engine's glibc-2.28 wheels. This
adapter preserves Evo2's checkpoint and architecture while running BF16 state
in FP32 on Turing GPUs and attention through the unfused PyTorch path. Existing
FP32 long-filter state remains FP32. The exact
compatibility mode is recorded in every cache manifest and must pass a real GPU
smoke before formal benchmark inclusion.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_public_dna_embeddings import (  # noqa: E402
    labels_array,
    read_dna,
    read_samples,
    reverse_complement,
    sha256_path,
    verified_sha256_path,
)


def masked_mean_pool(hidden, lengths):
    """Mean-pool [batch, sequence, hidden] arrays over non-padding positions."""
    values = np.asarray(hidden)
    if values.ndim != 3 or len(lengths) != values.shape[0]:
        raise ValueError("hidden must be [batch, sequence, hidden] with one length per row")
    pooled = []
    for row, length in zip(values, lengths):
        length = int(length)
        if length < 1 or length > row.shape[0]:
            raise ValueError(f"invalid pooling length {length} for sequence width {row.shape[0]}")
        pooled.append(row[:length].mean(axis=0))
    return np.asarray(pooled, dtype=np.float32)


def cast_bfloat16_state_to_float32(model):
    """Convert only BF16 parameters/buffers; preserve FP32 numerical state."""
    parameter_tensors = 0
    buffer_tensors = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dtype == torch.bfloat16:
                parameter.data = parameter.data.to(dtype=torch.float32)
                parameter_tensors += 1
        for module in model.modules():
            for name, buffer in list(module._buffers.items()):
                if buffer is not None and buffer.dtype == torch.bfloat16:
                    module._buffers[name] = buffer.to(dtype=torch.float32)
                    buffer_tensors += 1
    remaining = [
        name for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.dtype == torch.bfloat16
    ]
    if remaining:
        raise RuntimeError(f"Evo2 BF16 state remained after Turing conversion: {remaining[:5]}")
    return {"parameter_tensors": parameter_tensors, "buffer_tensors": buffer_tensors}


def load_evo2(checkpoint, config_path):
    from vortex.model.model import StripedHyena
    from vortex.model.tokenizer import CharLevelTokenizer
    from vortex.model.utils import dotdict, load_checkpoint

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if raw.get("model_name") != "shc-evo2-1b-8k-2T-v2" or raw.get("max_seqlen") != 8192:
        raise RuntimeError("unexpected Evo2 1B/8K runtime config")
    official_flags = {
        "use_fp8_input_projections": bool(raw.get("use_fp8_input_projections")),
        "use_flash_attn": bool(raw.get("use_flash_attn")),
    }
    raw["use_fp8_input_projections"] = False
    raw["use_flash_attn"] = False
    raw["max_batch_size"] = 1
    raw["params_dtype"] = torch.float32
    raw["attn_block_dtype"] = torch.float32
    raw["hyena_block_dtype"] = torch.float32
    raw["mlp_dtype"] = torch.float32
    config = dotdict(raw)
    model = StripedHyena(config)
    load_checkpoint(model, str(checkpoint))
    dtype_conversion = cast_bfloat16_state_to_float32(model)
    model.eval()
    tokenizer = CharLevelTokenizer(config.vocab_size)
    return model, tokenizer, config, official_flags, dtype_conversion


def embed_sequence(model, tokenizer, sequence, max_tokens):
    tokens = torch.as_tensor(tokenizer.tokenize(sequence), dtype=torch.long)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError(f"Evo2 tokenizer must return one token sequence, got shape={tuple(tokens.shape)}")
    if tokens.shape[1] > max_tokens:
        raise RuntimeError(
            f"Evo2 tokenizer would truncate sequence: tokens={tokens.shape[1]} max_tokens={max_tokens}"
        )
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    captured = {}

    def capture_norm(_module, _inputs, output):
        captured["hidden"] = output.detach()

    hook = model.norm.register_forward_hook(capture_norm)
    try:
        with torch.inference_mode():
            model(tokens)
    finally:
        hook.remove()
    if "hidden" not in captured:
        raise RuntimeError("Evo2 final norm hook produced no hidden representation")
    hidden = captured["hidden"].float().cpu().numpy()
    return masked_mean_pool(hidden, [tokens.shape[1]])[0], int(tokens.shape[1])


def extract_task(model, tokenizer, samples, context, max_tokens, include_rc_test, allow_shorter):
    sequence_cache = {}
    embeddings = []
    rc_embeddings = []
    rc_sample_ids = []
    token_lengths = []
    rc_token_lengths = []
    started = time.time()
    for row in samples:
        sequence = read_dna(row, sequence_cache, context, allow_shorter=allow_shorter)
        embedding, token_length = embed_sequence(model, tokenizer, sequence, max_tokens)
        embeddings.append(embedding)
        token_lengths.append(token_length)
        if include_rc_test and row["split"] == "test":
            rc_embedding, rc_token_length = embed_sequence(
                model, tokenizer, reverse_complement(sequence), max_tokens
            )
            rc_embeddings.append(rc_embedding)
            rc_token_lengths.append(rc_token_length)
            rc_sample_ids.append(row["sample_id"])
    matrix = np.asarray(embeddings, dtype=np.float16)
    rc_matrix = (
        np.asarray(rc_embeddings, dtype=np.float16)
        if rc_embeddings
        else np.empty((0, matrix.shape[1]), dtype=np.float16)
    )
    audit = {
        "samples": len(token_lengths),
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "mean_tokens": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "truncated_samples": 0,
    }
    rc_audit = {
        "samples": len(rc_token_lengths),
        "min_tokens": min(rc_token_lengths) if rc_token_lengths else 0,
        "max_tokens": max(rc_token_lengths) if rc_token_lengths else 0,
        "mean_tokens": float(np.mean(rc_token_lengths)) if rc_token_lengths else 0.0,
        "truncated_samples": 0,
    }
    return matrix, rc_matrix, np.asarray(rc_sample_ids), time.time() - started, audit, rc_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-verification-cache")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-rc-test", action="store_true")
    parser.add_argument("--require-no-truncation", action="store_true")
    parser.add_argument("--allow-shorter", action="store_true")
    parser.add_argument("--task", action="append", help="Only extract the named task; repeat for multiple tasks")
    args = parser.parse_args()

    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Evo2 extraction requires CUDA")
    if args.batch_size != 1:
        raise SystemExit("Evo2 1B locked runtime requires --batch-size 1")
    checkpoint = Path(args.checkpoint).resolve()
    try:
        actual_checkpoint_sha256 = verified_sha256_path(
            checkpoint, args.checkpoint_sha256, args.checkpoint_verification_cache,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    runtime_config = Path(args.runtime_config).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    out_dir = output_root / args.model_id / f"context_{args.context}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_config, official_flags, dtype_conversion = load_evo2(checkpoint, runtime_config)
    torch.cuda.reset_peak_memory_stats()
    task_manifests = {}
    total_started = time.time()
    samples_paths = sorted(dataset_root.glob("*/samples.tsv"))
    if args.task:
        requested_tasks = set(args.task)
        samples_paths = [path for path in samples_paths if path.parent.name in requested_tasks]
        found_tasks = {path.parent.name for path in samples_paths}
        if found_tasks != requested_tasks:
            raise SystemExit(f"requested tasks not found: {sorted(requested_tasks - found_tasks)}")
    for samples_path in samples_paths:
        task_id = samples_path.parent.name
        samples = read_samples(samples_path)
        embeddings, rc_embeddings, rc_sample_ids, elapsed, token_audit, rc_token_audit = extract_task(
            model, tokenizer, samples, args.context, args.max_tokens,
            args.include_rc_test, args.allow_shorter,
        )
        output_path = out_dir / f"{task_id}.npz"
        np.savez(
            output_path,
            embeddings=embeddings,
            sample_ids=np.asarray([row["sample_id"] for row in samples]),
            labels=labels_array(samples),
            splits=np.asarray([row["split"] for row in samples]),
            species=np.asarray([row["species"] for row in samples]),
            assemblies=np.asarray([row.get("assembly_id") or row["species"] for row in samples]),
            group_ids=np.asarray([row.get("group_id") or row.get("assembly_id") or row["species"] for row in samples]),
            rc_embeddings=rc_embeddings,
            rc_sample_ids=rc_sample_ids,
        )
        task_manifests[task_id] = {
            "samples": len(samples),
            "embedding_dim": int(embeddings.shape[1]),
            "rc_test_samples": int(len(rc_sample_ids)),
            "token_length_audit": token_audit,
            "rc_token_length_audit": rc_token_audit,
            "elapsed_seconds": elapsed,
            "samples_sha256": sha256_path(samples_path),
            "cache_path": str(output_path),
            "cache_sha256": sha256_path(output_path),
        }
        print(json.dumps({"status": "task_ok", "model_id": args.model_id, "task_id": task_id, **task_manifests[task_id]}), flush=True)

    manifest = {
        "status": "ok",
        "benchmark_id": "Plant-Genomic-Benchmark-publication-v2",
        "mode": "formal_evo2_frozen_embedding_cache",
        "model_id": args.model_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": sha256_path(runtime_config),
        "implementation_sha256": sha256_path(Path(__file__).resolve()),
        "official_runtime_flags": official_flags,
        "compatibility_mode": {
            "use_fp8_input_projections": False,
            "use_flash_attn": False,
            "architecture_unchanged": True,
            "checkpoint_parameters_unchanged_except_dtype_cast": True,
            "bfloat16_state_cast_to": "float32",
            "float32_state_preserved": True,
            "dtype_conversion": dtype_conversion,
            "reason": "CentOS7 glibc2.17 cannot load official glibc2.28 Transformer Engine wheels; RTX 2080 Ti sm75 cannot execute BF16 Triton kernels",
            "requires_real_gpu_smoke_before_formal_inclusion": True,
        },
        "pooling": "mean_of_final_rmsnorm_hidden_states",
        "dataset_root": str(dataset_root),
        "context": args.context,
        "max_tokens": args.max_tokens,
        "max_model_context": int(model_config.max_seqlen),
        "require_no_truncation": args.require_no_truncation,
        "allow_shorter": args.allow_shorter,
        "device": "cuda",
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "tasks": task_manifests,
        "elapsed_seconds": time.time() - total_started,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
    }
    manifest_path = out_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
