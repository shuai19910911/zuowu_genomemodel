#!/usr/bin/env python3
"""Extract Hugging Face DNA-model embeddings for formal CropGenome-Bench v1."""

import argparse
import csv
import fcntl
import hashlib
import importlib
import inspect
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_server_transfer.downstream_v4.data import public_model_snapshot_sha256
from training_server_transfer.downstream_v4.model_adapters import clean_model_runtime_cache, register_model_plugin
from training_server_transfer.downstream_v4.streaming_embeddings import (
    DiskBackedEmbeddingAccumulator,
)

BASES = np.asarray(list("ACGTN"))
RC_TABLE = str.maketrans("ACGTN", "TGCAN")


def sha256_path(path: Path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verified_sha256_path(path, expected_sha256, cache_path=None):
    path = Path(path).resolve()
    if cache_path is None:
        actual = sha256_path(path)
        if actual != expected_sha256:
            raise RuntimeError(f"SHA256 mismatch: {path}")
        return actual
    cache_path = Path(cache_path).resolve(); cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            stat = path.stat()
            try:
                record = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {}
            if (
                record.get("path") == str(path)
                and record.get("sha256") == expected_sha256
                and int(record.get("size_bytes", -1)) == stat.st_size
                and int(record.get("mtime_ns", -1)) == stat.st_mtime_ns
            ):
                return expected_sha256
            actual = sha256_path(path)
            if actual != expected_sha256:
                raise RuntimeError(f"SHA256 mismatch: {path}")
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"file changed while hashing: {path}")
            payload = {
                "path": str(path), "size_bytes": after.st_size,
                "mtime_ns": after.st_mtime_ns, "sha256": actual,
                "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, cache_path)
            return actual
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def parse_label(row):
    if "label_json" in row and row["label_json"] != "":
        value = json.loads(row["label_json"])
        if isinstance(value, list):
            return [float(item) for item in value]
        return int(value) if isinstance(value, int) else float(value)
    return int(row["label"])


def labels_array(samples):
    values = [row["label"] for row in samples]
    if values and all(isinstance(value, int) for value in values):
        return np.asarray(values, dtype=np.int8)
    return np.asarray(values, dtype=np.float32)


EXPECTED_TASK_CACHE_FIELDS = {
    "embeddings", "sample_ids", "labels", "splits", "species", "assemblies",
    "group_ids", "rc_embeddings", "rc_sample_ids",
}


def validate_existing_task_cache(path, samples, include_rc_test):
    """Validate a partial task cache strongly enough to resume without recompute."""
    expected_sample_ids = np.asarray([row["sample_id"] for row in samples])
    expected_splits = np.asarray([row["split"] for row in samples])
    expected_species = np.asarray([row["species"] for row in samples])
    expected_assemblies = np.asarray([
        row.get("assembly_id") or row["species"] for row in samples
    ])
    expected_rc_ids = np.asarray([
        row["sample_id"] for row in samples if row["split"] == "test"
    ])
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != EXPECTED_TASK_CACHE_FIELDS:
                raise RuntimeError("task cache field set is invalid")
            exact_fields = {
                "sample_ids": expected_sample_ids,
                "labels": labels_array(samples),
                "splits": expected_splits,
                "species": expected_species,
                "assemblies": expected_assemblies,
                "group_ids": np.asarray([
                    row.get("group_id") or row.get("assembly_id") or row["species"]
                    for row in samples
                ]),
                "rc_sample_ids": expected_rc_ids if include_rc_test else np.asarray([]),
            }
            for name, expected in exact_fields.items():
                observed = payload[name]
                equal = (
                    np.array_equal(observed, expected, equal_nan=True)
                    if np.issubdtype(observed.dtype, np.number)
                    and np.issubdtype(expected.dtype, np.number)
                    else np.array_equal(observed, expected)
                )
                if not equal:
                    raise RuntimeError(f"task cache {name} does not match frozen dataset")
            embeddings = payload["embeddings"]
            rc_embeddings = payload["rc_embeddings"]
            if (
                embeddings.ndim != 2 or embeddings.shape[0] != len(samples)
                or embeddings.shape[1] <= 0
            ):
                raise RuntimeError("task cache embeddings shape is invalid")
            if (
                rc_embeddings.ndim != 2
                or rc_embeddings.shape != (len(exact_fields["rc_sample_ids"]), embeddings.shape[1])
            ):
                raise RuntimeError("task cache rc_embeddings shape is invalid")
            require_finite_embeddings(embeddings, "existing forward")
            require_finite_embeddings(rc_embeddings, "existing reverse_complement")
            return {
                "embedding_dim": int(embeddings.shape[1]),
                "rc_test_samples": int(rc_embeddings.shape[0]),
            }
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"cannot validate existing task cache {path}: {error}") from error


def read_samples(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for row in rows:
        input_path = Path(row["input_path"])
        if not input_path.is_absolute():
            input_path = path.parent / input_path
        row["input_path"] = str(input_path.resolve())
        row["offset"] = int(row["offset"])
        row["length"] = int(row["length"])
        row["label"] = parse_label(row)
    return rows


def read_dna(row, cache, context, allow_shorter=False):
    path = row["input_path"]
    if path not in cache:
        cache[path] = np.memmap(path, dtype=np.uint8, mode="r")
    tokens = np.asarray(cache[path][row["offset"]:row["offset"] + row["length"]], dtype=np.uint8)
    if context > len(tokens) and not allow_shorter:
        raise ValueError(f"context {context} exceeds sample length {len(tokens)}")
    if context < len(tokens):
        start = (len(tokens) - context) // 2
        tokens = tokens[start:start + context]
    safe = np.minimum(tokens, 4)
    return "".join(BASES[safe].tolist())


def reverse_complement(sequence):
    return sequence.translate(RC_TABLE)[::-1]


def model_hidden(outputs):
    if getattr(outputs, "last_hidden_state", None) is not None:
        return outputs.last_hidden_state
    if getattr(outputs, "hidden_states", None):
        return outputs.hidden_states[-1]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    raise RuntimeError("model output has no hidden representation")


def canonicalize_rcps_hidden(hidden, d_model, rcps):
    """Collapse aligned RCPS channel pairs before masked sequence pooling.

    Caduceus stores the RC half aligned to the original sequence positions. A
    sequence flip is part of the full hidden-state RC operation, but cancels
    with the correspondingly flipped attention mask during mean pooling. Only
    the channel axis is therefore reversed here; flipping sequence positions
    would misalign asymmetric special-token masks such as Caduceus' trailing
    SEP token.
    """
    if not rcps:
        return hidden
    if d_model is None or hidden.shape[-1] != 2 * int(d_model):
        raise RuntimeError(
            f"RCPS hidden width mismatch: width={hidden.shape[-1]} d_model={d_model}"
        )
    d_model = int(d_model)
    forward = hidden[..., :d_model]
    reverse = hidden[..., d_model:]
    if torch.is_tensor(hidden):
        reverse = torch.flip(reverse, dims=(-1,))
    else:
        reverse = np.flip(reverse, axis=-1)
    return 0.5 * (forward + reverse)


def load_hf_model(model_ref, revision, trust_remote_code, dtype, local_files_only, model_head):
    clean_model_runtime_cache(model_ref)
    plugin = register_model_plugin(model_ref)
    if plugin.force_trust_remote_code is not None:
        trust_remote_code = plugin.force_trust_remote_code
    common = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    tokenizer = plugin.tokenizer or AutoTokenizer.from_pretrained(model_ref, **common)
    model_kwargs = {**common, "torch_dtype": dtype, "low_cpu_mem_usage": True}
    config = None
    if plugin.config_overrides:
        config = AutoConfig.from_pretrained(model_ref, **common)
        for key, value in plugin.config_overrides.items():
            setattr(config, key, value)
    if model_head == "masked-lm-base":
        wrapper = AutoModelForMaskedLM.from_pretrained(
            model_ref,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            config=config,
            **common,
        )
        backbone = getattr(wrapper, "esm", None) or getattr(wrapper, "plantbimoe", None)
        if backbone is None:
            raise RuntimeError("masked-LM model does not expose a supported base backbone")
        return tokenizer, backbone, f"{plugin.mode}+masked_lm_base_backbone"

    try:
        if config is not None:
            model_kwargs["config"] = config
        model = AutoModel.from_pretrained(model_ref, **model_kwargs)
        if model.__class__.__name__.endswith("ForMaskedLM"):
            for attribute in ("bert", "esm", "plantbimoe"):
                backbone = getattr(model, attribute, None)
                if backbone is not None:
                    return tokenizer, backbone, f"{plugin.mode}+auto_mlm_{attribute}_backbone"
            raise RuntimeError("AutoModel resolved to a masked-LM wrapper without a supported base backbone")
        return tokenizer, model, f"{plugin.mode}+auto_model"
    except ValueError as exc:
        if not trust_remote_code or "config_class" not in str(exc):
            raise
        config = AutoConfig.from_pretrained(model_ref, **common)
        model_class_ref = (getattr(config, "auto_map", None) or {}).get("AutoModel")
        if not model_class_ref:
            raise
        model_class = get_class_from_dynamic_module(
            model_class_ref,
            model_ref,
            revision=revision,
            local_files_only=local_files_only,
        )
        model_class.config_class = config.__class__
        fallback_kwargs = {**model_kwargs, "low_cpu_mem_usage": False, "add_pooling_layer": False}
        model = model_class.from_pretrained(model_ref, config=config, **fallback_kwargs)
        return tokenizer, model, f"{plugin.mode}+dynamic_config_class_binding"


def token_length_audit(tokenizer, sequences, max_tokens):
    encoded = tokenizer(
        sequences,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    lengths = [len(values) for values in input_ids]
    return {
        "samples": len(lengths),
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "sum_tokens": sum(lengths),
        "truncated_samples": sum(length > max_tokens for length in lengths),
    }


def audit_existing_task_tokens(
    tokenizer, samples, context, max_tokens, include_rc_test, allow_shorter,
    batch_size=256,
):
    cache = {}
    totals = [
        {"samples": 0, "min_tokens": None, "max_tokens": 0, "sum_tokens": 0, "truncated_samples": 0},
        {"samples": 0, "min_tokens": None, "max_tokens": 0, "sum_tokens": 0, "truncated_samples": 0},
    ]

    def update(total, audit):
        total["samples"] += audit["samples"]
        total["sum_tokens"] += audit["sum_tokens"]
        total["truncated_samples"] += audit["truncated_samples"]
        total["max_tokens"] = max(total["max_tokens"], audit["max_tokens"])
        if audit["samples"]:
            total["min_tokens"] = (
                audit["min_tokens"] if total["min_tokens"] is None
                else min(total["min_tokens"], audit["min_tokens"])
            )

    for start in range(0, len(samples), batch_size):
        rows = samples[start:start + batch_size]
        sequences = [
            read_dna(row, cache, context, allow_shorter=allow_shorter) for row in rows
        ]
        update(totals[0], token_length_audit(tokenizer, sequences, max_tokens))
        if include_rc_test:
            rc_sequences = [
                reverse_complement(sequence)
                for row, sequence in zip(rows, sequences) if row["split"] == "test"
            ]
            if rc_sequences:
                update(totals[1], token_length_audit(tokenizer, rc_sequences, max_tokens))
    for total in totals:
        total["min_tokens"] = total["min_tokens"] or 0
        total["mean_tokens"] = (
            total["sum_tokens"] / total["samples"] if total["samples"] else 0.0
        )
    return totals


def resolve_inference_dtype(name, device):
    if device.type != "cuda":
        return torch.float32
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return dtypes[name]
    except KeyError as error:
        raise ValueError(f"unsupported inference dtype: {name!r}") from error


def require_finite_embeddings(values, label):
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite public-model embeddings detected: {label}")


def supported_forward_kwargs(model, kwargs):
    parameters = inspect.signature(model.forward).parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    supported = {parameter.name for parameter in parameters}
    return {key: value for key, value in kwargs.items() if key in supported}


def embed_strings(
    model, tokenizer, sequences, device, max_tokens, inference_dtype=torch.bfloat16,
):
    encoded = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_special_tokens_mask=True,
    )
    special_mask = encoded.pop("special_tokens_mask", None)
    input_ids = encoded.get("input_ids")
    if input_ids is not None and (special_mask is None or special_mask.shape != input_ids.shape):
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_token_id in tokenizer.all_special_ids:
            special_mask |= input_ids.eq(int(special_token_id))
    encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
    forward_kwargs = supported_forward_kwargs(
        model,
        {**encoded, "output_hidden_states": False, "return_dict": True},
    )
    autocast_enabled = device.type == "cuda" and inference_dtype in {
        torch.bfloat16, torch.float16,
    }
    autocast_dtype = inference_dtype if autocast_enabled else torch.float16
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled,
    ):
        outputs = model(**forward_kwargs)
        hidden = model_hidden(outputs)
        hidden = canonicalize_rcps_hidden(
            hidden,
            d_model=getattr(model.config, "d_model", None),
            rcps=bool(getattr(model.config, "rcps", False)),
        )
        pool_mask = encoded.get("attention_mask", torch.ones(hidden.shape[:2], dtype=torch.long, device=device)).bool()
        if special_mask is not None:
            pool_mask = pool_mask & ~special_mask.to(device).bool()
        weights = pool_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return pooled.to(dtype=torch.float16).cpu().numpy()


def extract_task(
    model, tokenizer, samples, batch_size, context, max_tokens, device,
    include_rc_test, require_no_truncation, allow_shorter=False,
    inference_dtype=torch.bfloat16, workspace=None,
):
    cache = {}
    rc_sample_ids = []
    rc_rows = sum(
        row["split"] == "test" for row in samples
    ) if include_rc_test else 0
    sink = DiskBackedEmbeddingAccumulator(
        workspace, forward_rows=len(samples), rc_rows=rc_rows,
    )
    length_stats = {"samples": 0, "min_tokens": None, "max_tokens": 0, "sum_tokens": 0, "truncated_samples": 0}
    rc_length_stats = {"samples": 0, "min_tokens": None, "max_tokens": 0, "sum_tokens": 0, "truncated_samples": 0}

    def update_length_stats(total, batch):
        total["samples"] += batch["samples"]
        total["sum_tokens"] += batch["sum_tokens"]
        total["truncated_samples"] += batch["truncated_samples"]
        total["max_tokens"] = max(total["max_tokens"], batch["max_tokens"])
        if batch["samples"]:
            total["min_tokens"] = batch["min_tokens"] if total["min_tokens"] is None else min(total["min_tokens"], batch["min_tokens"])

    started = time.time()
    try:
        for start in range(0, len(samples), batch_size):
            batch_rows = samples[start:start + batch_size]
            sequences = [read_dna(row, cache, context, allow_shorter=allow_shorter) for row in batch_rows]
            audit = token_length_audit(tokenizer, sequences, max_tokens)
            update_length_stats(length_stats, audit)
            if require_no_truncation and audit["truncated_samples"]:
                raise RuntimeError(
                    f"tokenizer would truncate {audit['truncated_samples']} samples in batch; "
                    f"max_observed_tokens={audit['max_tokens']} max_tokens={max_tokens} context={context}"
                )
            forward = embed_strings(
                model, tokenizer, sequences, device, max_tokens, inference_dtype,
            )
            reverse = None
            if include_rc_test:
                test_rows = [(row, sequence) for row, sequence in zip(batch_rows, sequences) if row["split"] == "test"]
                if test_rows:
                    rc_sequences = [reverse_complement(sequence) for _, sequence in test_rows]
                    rc_audit = token_length_audit(tokenizer, rc_sequences, max_tokens)
                    update_length_stats(rc_length_stats, rc_audit)
                    if require_no_truncation and rc_audit["truncated_samples"]:
                        raise RuntimeError(
                            f"tokenizer would truncate {rc_audit['truncated_samples']} RC samples in batch; "
                            f"max_observed_tokens={rc_audit['max_tokens']} max_tokens={max_tokens} context={context}"
                        )
                    reverse = embed_strings(
                        model, tokenizer, rc_sequences, device, max_tokens, inference_dtype,
                    )
                    rc_sample_ids.extend(row["sample_id"] for row, _ in test_rows)
            sink.append(forward, reverse)
    except Exception:
        sink.cleanup()
        raise
    for stats in (length_stats, rc_length_stats):
        stats["min_tokens"] = stats["min_tokens"] or 0
        stats["mean_tokens"] = stats["sum_tokens"] / stats["samples"] if stats["samples"] else 0.0
    return sink, np.asarray(rc_sample_ids), time.time() - started, length_stats, rc_length_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--weight-file", required=True)
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--weight-verification-cache")
    parser.add_argument("--model-snapshot-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inference-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--load-dtype",
        choices=["bfloat16", "float16", "float32"],
        help="Optional construction/loading dtype; model is converted to --inference-dtype afterwards",
    )
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-head", choices=["auto", "masked-lm-base"], default="auto")
    parser.add_argument("--disable-remote-triton", action="store_true")
    parser.add_argument("--include-rc-test", action="store_true")
    parser.add_argument("--require-no-truncation", action="store_true")
    parser.add_argument("--allow-shorter", action="store_true", help="Use a full sequence when shorter than --context")
    parser.add_argument("--task", action="append", help="Only extract the named task; repeat for multiple tasks")
    parser.add_argument(
        "--resume-valid-task-caches", action="store_true",
        help="Reuse structurally exact finite task NPZ files after re-auditing token lengths",
    )
    args = parser.parse_args()

    hf_model = Path(args.hf_model).resolve()
    clean_model_runtime_cache(hf_model)
    weight_path = (hf_model / args.weight_file).resolve()
    if weight_path.parent != hf_model and hf_model not in weight_path.parents:
        raise SystemExit("public model weight path escapes --hf-model")
    if not weight_path.is_file():
        raise SystemExit(f"public model weight file is missing: {weight_path}")
    try:
        actual_weight_sha256 = verified_sha256_path(
            weight_path, args.weight_sha256, args.weight_verification_cache,
        )
    except RuntimeError as error:
        raise SystemExit(f"public model weight SHA256 mismatch: {weight_path}") from error
    actual_model_snapshot_sha256 = public_model_snapshot_sha256(hf_model, args.weight_file)
    if actual_model_snapshot_sha256 != args.model_snapshot_sha256:
        raise SystemExit("public model non-weight snapshot SHA256 mismatch")
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    out_dir = output_root / args.model_id / f"context_{args.context}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise SystemExit("CUDA requested but unavailable")
    dtype = resolve_inference_dtype(args.inference_dtype, device)
    load_dtype = resolve_inference_dtype(args.load_dtype, device) if args.load_dtype else dtype
    tokenizer, model, load_compatibility_mode = load_hf_model(
        str(hf_model),
        args.revision,
        args.trust_remote_code,
        load_dtype,
        args.local_files_only,
        args.model_head,
    )
    if args.disable_remote_triton:
        model_module = importlib.import_module(model.__class__.__module__)
        if not hasattr(model_module, "flash_attn_qkvpacked_func"):
            raise SystemExit("model module has no flash_attn_qkvpacked_func fallback switch")
        model_module.flash_attn_qkvpacked_func = None
    model = model.eval().to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    task_manifests = {}
    total_started = time.time()
    effective_batch_size = args.batch_size
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
        output_path = out_dir / f"{task_id}.npz"
        if args.resume_valid_task_caches and output_path.is_file():
            cache_shape = validate_existing_task_cache(
                output_path, samples, include_rc_test=args.include_rc_test,
            )
            token_audit, rc_token_audit = audit_existing_task_tokens(
                tokenizer, samples, args.context, args.max_tokens,
                args.include_rc_test, args.allow_shorter,
            )
            if args.require_no_truncation and (
                token_audit["truncated_samples"] or rc_token_audit["truncated_samples"]
            ):
                raise RuntimeError(f"existing task cache token audit failed for {task_id}")
            task_manifests[task_id] = {
                "samples": len(samples),
                **cache_shape,
                "token_length_audit": token_audit,
                "rc_token_length_audit": rc_token_audit,
                "elapsed_seconds": 0.0,
                "samples_sha256": sha256_path(samples_path),
                "cache_path": str(output_path),
                "cache_sha256": sha256_path(output_path),
                "reused_existing_cache": True,
            }
            print(json.dumps({
                "status": "task_reused", "model_id": args.model_id,
                "task_id": task_id, **task_manifests[task_id],
            }, ensure_ascii=False), flush=True)
            continue
        while True:
            try:
                sink, rc_sample_ids, elapsed, token_audit, rc_token_audit = extract_task(
                    model, tokenizer, samples, effective_batch_size, args.context, args.max_tokens, device,
                    args.include_rc_test, args.require_no_truncation,
                    allow_shorter=args.allow_shorter, inference_dtype=dtype,
                    workspace=output_path.with_name(output_path.name + ".parts"),
                )
                break
            except torch.cuda.OutOfMemoryError:
                if effective_batch_size <= args.min_batch_size:
                    raise
                effective_batch_size = max(args.min_batch_size, effective_batch_size // 2)
                torch.cuda.empty_cache()
                print(json.dumps({"status": "oom_retry", "task_id": task_id, "new_batch_size": effective_batch_size}), flush=True)
        embedding_dim = sink.embedding_dim
        try:
            sink.save_npz(
                output_path,
            sample_ids=np.asarray([row["sample_id"] for row in samples]),
            labels=labels_array(samples),
            splits=np.asarray([row["split"] for row in samples]),
            species=np.asarray([row["species"] for row in samples]),
            assemblies=np.asarray([row.get("assembly_id") or row["species"] for row in samples]),
            group_ids=np.asarray([row.get("group_id") or row.get("assembly_id") or row["species"] for row in samples]),
            rc_sample_ids=rc_sample_ids,
            )
        except Exception:
            sink.cleanup()
            raise
        task_manifests[task_id] = {
            "samples": len(samples),
            "embedding_dim": int(embedding_dim),
            "rc_test_samples": int(len(rc_sample_ids)),
            "token_length_audit": token_audit,
            "rc_token_length_audit": rc_token_audit,
            "elapsed_seconds": elapsed,
            "samples_sha256": sha256_path(samples_path),
            "cache_path": str(output_path),
            "cache_sha256": sha256_path(output_path),
        }
        print(json.dumps({"status": "task_ok", "model_id": args.model_id, "task_id": task_id, **task_manifests[task_id]}, ensure_ascii=False), flush=True)

    manifest = {
        "status": "ok",
        "benchmark_id": "CropGenome-Bench-v1",
        "mode": "formal_public_model_frozen_embedding_cache",
        "model_id": args.model_id,
        "hf_model": args.hf_model,
        "weight_file": args.weight_file,
        "weight_sha256": actual_weight_sha256,
        "model_snapshot_sha256": actual_model_snapshot_sha256,
        "implementation_sha256": sha256_path(Path(__file__).resolve()),
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "model_type": getattr(model.config, "model_type", None),
        "architectures": getattr(model.config, "architectures", None),
        "rcps_canonicalized": bool(getattr(model.config, "rcps", False)),
        "backbone_d_model": getattr(model.config, "d_model", None),
        "load_compatibility_mode": load_compatibility_mode,
        "model_head": args.model_head,
        "trust_remote_code": args.trust_remote_code,
        "remote_triton_disabled": args.disable_remote_triton,
        "dataset_root": str(dataset_root),
        "context": args.context,
        "max_tokens": args.max_tokens,
        "require_no_truncation": args.require_no_truncation,
        "allow_shorter": args.allow_shorter,
        "requested_batch_size": args.batch_size,
        "effective_batch_size": effective_batch_size,
        "device": str(device),
        "inference_dtype": str(dtype).removeprefix("torch."),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "tasks": task_manifests,
        "resume_valid_task_caches": args.resume_valid_task_caches,
        "elapsed_seconds": time.time() - total_started,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
    }
    (out_dir / "cache_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "manifest": manifest}, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
