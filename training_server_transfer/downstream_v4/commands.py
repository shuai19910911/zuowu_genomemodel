"""Build fully explicit, hash-bound encode and zero-shot commands."""

import importlib.util
import json
import sys
from pathlib import Path

from .data import sha256_path
from .model_adapters import clean_model_runtime_cache, runtime_spec
from .registry import FORBIDDEN_INTERNAL_MODEL_IDS


PYTHON = "/home/user/zhangzhishuai/.local/share/mamba/envs/zuowu_genomemodel/bin/python"


def _model(registry, model_id):
    return next(row for row in registry["models"] if row["model_id"] == model_id)


def _task(registry, task_id):
    return next(row for row in registry["tasks"] if row["task_id"] == task_id)


def _scaled_public_batch_size(spec, context, maximum=32):
    """Reuse the profiled full-context token budget at shorter contexts."""
    context = max(1, int(context))
    base_batch = int(spec["batch_size"])
    profiled_context = int(spec["context_bp"])
    scale = max(1, profiled_context // context)
    return min(int(maximum), base_batch * scale)


def _cropgenome_batch_size(context):
    context = int(context)
    if context <= 1024:
        return 16
    if context <= 4096:
        return 8
    return 1


def _load_snapshot_hash_function(project_root):
    path = Path(project_root) / "scripts/publication_v2_contracts.py"
    spec = importlib.util.spec_from_file_location("cropgenome_v4_publication_contracts", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.public_model_snapshot_sha256


def _public_weight_file(model_root):
    model_root = Path(model_root)
    receipt_path = model_root / "download_receipt.json"
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid model download receipt: {receipt_path}: {error}") from error
        selected = [
            model_root / relative for relative in receipt.get("selected_weight_files") or []
            if not str(relative).endswith(".index.json")
        ]
        if selected:
            if len(selected) != 1:
                raise ValueError(f"runtime currently requires one selected weight file, found {len(selected)}")
            if not selected[0].is_file():
                raise FileNotFoundError(selected[0])
            return selected[0]
    candidates = [
        path for path in model_root.iterdir()
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".pt", ".pth"}
    ]
    if not candidates:
        raise FileNotFoundError(f"no public model weight file at root of {model_root}")
    return max(candidates, key=lambda path: path.stat().st_size)


def validate_model_receipt(model, model_root):
    receipt_path = Path(model_root) / "download_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid model receipt for {model['model_id']}: {error}") from error
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "ok"
        or receipt.get("model_id") != model["model_id"]
        or receipt.get("model_ref") != model["model_ref"]
        or receipt.get("revision") != model["revision"]
    ):
        raise ValueError(f"model receipt does not match frozen registry: {model['model_id']}")
    return receipt


def build_encode_command(registry, task_id, model_id, project_root, dataset_root, output_root,
                         checkpoint=None, model_config=None, context=None, device="cuda"):
    project_root = Path(project_root).resolve()
    # Keep the lexical path: grouped final runs use a canonical symlink farm.
    # Resolving the first task symlink would redirect --dataset-root to that
    # task's original parent and make every sibling task disappear.
    dataset_root = Path(dataset_root).absolute()
    output_root = Path(output_root).resolve()
    task = _task(registry, task_id); model = _model(registry, model_id)
    if model_id in FORBIDDEN_INTERNAL_MODEL_IDS or model.get("kind") == "internal_control":
        raise ValueError(f"internal pretraining ablation is disabled by user: {model_id}")
    dataset_parent = dataset_root.parent if (dataset_root / "samples.tsv").is_file() else dataset_root
    if model["kind"] == "simple_baseline":
        if task["task_kind"].startswith("zero_shot") or task["task_kind"] in {"token_multiclass", "sensitivity_track"}:
            raise ValueError(f"{model_id} is not applicable to {task['task_kind']}")
        chosen_context = int(context or max(task.get("context_bp") or [512]))
        output_path = output_root / model_id / f"context_{chosen_context}" / f"{task_id}.npz"
        return [
            PYTHON, str(project_root / "scripts/run_cropgenome_downstream_v4.py"),
            "--project-root", str(project_root), "baseline", "--task-id", task_id,
            "--dataset-root", str(dataset_root), "--output", str(output_path),
            "--context", str(chosen_context),
        ]
    if model["kind"] in {"cropgenome", "project_model"}:
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint or "CropGenome checkpoint not supplied")
        config_reference = model.get(
            "model_config",
            model.get("model_ref", "training_server_transfer/configs/model_large.json"),
        )
        model_config = Path(model_config or project_root / config_reference).resolve()
        chosen_context = int(context or max(task.get("context_bp") or [512]))
        source_args = ["--checkpoint", str(Path(checkpoint).resolve())]
        if task_id == "B13":
            return [
                PYTHON, str(project_root / "scripts/extract_cropgenome_structure_token_embeddings.py"),
                "--project-root", str(project_root), "--samples", str(dataset_root / "samples.tsv"),
                *source_args, "--model-config", str(model_config),
                "--model-id", model_id, "--output-dir", str(output_root / model_id / f"context_{chosen_context}"),
                "--context", str(chosen_context), "--batch-size", "1", "--device", device,
            ]
        return [
            PYTHON, str(project_root / "scripts/extract_cropgenome_bench_v1_embeddings.py"),
            "--project-root", str(project_root), "--dataset-root", str(dataset_parent),
            "--model-config", str(model_config), *source_args,
            "--model-id", model_id, "--output-root", str(output_root),
            "--context", str(chosen_context),
            "--batch-size", str(_cropgenome_batch_size(chosen_context)),
            "--device", device, "--task", task_id, "--include-rc-test", "--allow-shorter",
            "--resume-valid-task-caches",
        ]
    if model["kind"] != "public_weight":
        raise ValueError(f"model has no runnable weight adapter: {model_id}")
    spec = runtime_spec(model_id)
    chosen_context = int(context or min(512, spec["context_bp"]))
    if chosen_context > spec["context_bp"]:
        raise ValueError(f"requested context {chosen_context} exceeds {model_id} runtime limit {spec['context_bp']}")
    model_root = project_root / "training_server_transfer/public_models/downstream_v4" / model_id
    download_receipt = validate_model_receipt(model, model_root)
    clean_model_runtime_cache(model_root)
    weight = _public_weight_file(model_root)
    weight_relative = weight.relative_to(model_root).as_posix()
    weight_record = next(
        (record for record in download_receipt.get("artifacts") or []
         if record.get("path") == weight_relative),
        None,
    )
    if not weight_record or not weight_record.get("sha256"):
        raise ValueError(f"download receipt has no frozen hash for {weight_relative}")
    if weight.stat().st_size != int(weight_record["size_bytes"]):
        raise ValueError(f"public model weight size changed: {weight}")
    weight_sha256 = weight_record["sha256"]
    verification_cache = (
        model_root.parent / ".verification" / f"{model_id}.weight.json"
    )
    if spec["model_head"] == "evo2":
        runtime_config = project_root / model["runtime_config"]
        if not runtime_config.is_file():
            raise FileNotFoundError(runtime_config)
        return [
            PYTHON, str(project_root / "scripts/extract_evo2_embeddings.py"),
            "--dataset-root", str(dataset_parent), "--checkpoint", str(weight),
            "--runtime-config", str(runtime_config), "--checkpoint-sha256", weight_sha256,
            "--checkpoint-verification-cache", str(verification_cache),
            "--model-id", model_id, "--output-root", str(output_root),
            "--context", str(chosen_context), "--max-tokens", str(min(chosen_context, spec["max_tokens"])),
            "--batch-size", "1", "--device", device, "--task", task_id,
            "--include-rc-test", "--require-no-truncation", "--allow-shorter",
        ]
    snapshot_hash = _load_snapshot_hash_function(project_root)(model_root, weight.name)
    if task_id == "B13":
        command = [
            PYTHON, str(project_root / "scripts/extract_public_token_embeddings.py"),
            "--samples", str(dataset_root / "samples.tsv"), "--hf-model", str(model_root),
            "--weight-file", weight.name, "--weight-sha256", weight_sha256,
            "--weight-verification-cache", str(verification_cache),
            "--model-snapshot-sha256", snapshot_hash, "--model-id", model_id,
            "--output-dir", str(output_root / model_id / f"context_{chosen_context}"),
            "--context", str(chosen_context), "--batch-size", str(spec["batch_size"]),
            "--device", device, "--model-head", spec["model_head"], "--local-files-only",
            "--inference-dtype", spec.get("inference_dtype", "float16"),
        ]
        if spec["trust_remote_code"]:
            command.append("--trust-remote-code")
        if spec.get("disable_remote_triton"):
            command.append("--disable-remote-triton")
        if spec.get("load_dtype"):
            command.extend(["--load-dtype", spec["load_dtype"]])
        return command
    command = [
        PYTHON, str(project_root / "scripts/extract_public_dna_embeddings.py"),
        "--dataset-root", str(dataset_parent), "--hf-model", str(model_root),
        "--weight-file", weight.name, "--weight-sha256", weight_sha256,
        "--weight-verification-cache", str(verification_cache),
        "--model-snapshot-sha256", snapshot_hash, "--model-id", model_id,
        "--output-root", str(output_root), "--context", str(chosen_context),
        "--max-tokens", str(spec["max_tokens"]),
        "--batch-size", str(_scaled_public_batch_size(spec, chosen_context)),
        "--min-batch-size", "1", "--device", device,
        "--inference-dtype", spec.get("inference_dtype", "float16"),
        "--model-head", spec["model_head"], "--local-files-only", "--task", task_id,
        "--include-rc-test", "--require-no-truncation", "--allow-shorter",
        "--resume-valid-task-caches",
    ]
    if spec["trust_remote_code"]:
        command.append("--trust-remote-code")
    if spec.get("disable_remote_triton"):
        command.append("--disable-remote-triton")
    if spec.get("load_dtype"):
        command.extend(["--load-dtype", spec["load_dtype"]])
    return command


def build_zero_shot_command(registry_path, task_id, model_id, project_root, dataset_root,
                            output_root, checkpoint=None, model_config=None, context=None,
                            device="cuda"):
    project_root = Path(project_root).resolve()
    command = [
        PYTHON, str(project_root / "scripts/run_cropgenome_downstream_v4.py"),
        "--registry", str(Path(registry_path).resolve()), "zero-shot",
        "--task-id", task_id, "--model-id", model_id,
        "--dataset-root", str(Path(dataset_root).resolve()),
        "--output-root", str(Path(output_root).resolve()), "--device", device,
    ]
    if checkpoint:
        command.extend(["--checkpoint", str(Path(checkpoint).resolve())])
    if model_config:
        command.extend(["--model-config", str(Path(model_config).resolve())])
    if context is not None:
        command.extend(["--context", str(int(context))])
    return command
