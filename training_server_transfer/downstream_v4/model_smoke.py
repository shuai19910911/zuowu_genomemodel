"""Real GPU forward smoke gate for every frozen public model adapter."""

import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .commands import build_encode_command
from .data import materialize_rows, sha256_path
from .gpu_gate import run_with_dynamic_gpu
from .model_adapters import runtime_spec


MODEL_SMOKE_PROTOCOL_ID = "public_model_real_gpu_forward_smoke_v3"
RUNTIME_CONTRACT_FIELDS = (
    "runtime_config", "runtime_pythonpath", "runtime_imports",
)


def _canonical_sha(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def smoke_implementation_paths(model, project_root):
    project_root = Path(project_root).resolve()
    implementation_paths = [
        Path(__file__).resolve(),
        project_root / "training_server_transfer/downstream_v4/commands.py",
        project_root / "training_server_transfer/downstream_v4/gpu_gate.py",
        project_root / "training_server_transfer/downstream_v4/model_adapters.py",
    ]
    runtime = runtime_spec(model["model_id"])
    if runtime.get("model_head") == "evo2":
        implementation_paths.append(project_root / "scripts/extract_evo2_embeddings.py")
    else:
        implementation_paths.append(project_root / "scripts/extract_public_dna_embeddings.py")
        if runtime.get("token_aligned"):
            implementation_paths.extend([
                project_root / "scripts/extract_public_token_embeddings.py",
                project_root / "scripts/extract_cropgenome_structure_token_embeddings.py",
            ])
    return implementation_paths


def model_runtime_contract(model):
    """Preserve generator-owned model-loader extensions in the smoke identity."""
    return {
        key: model[key] for key in RUNTIME_CONTRACT_FIELDS if key in model
    }


def smoke_contract_sha256(model, project_root):
    project_root = Path(project_root).resolve()
    implementation_paths = smoke_implementation_paths(model, project_root)
    runtime = runtime_spec(model["model_id"])
    payload = {
        "protocol_id": MODEL_SMOKE_PROTOCOL_ID,
        "model_id": model["model_id"], "revision": model["revision"],
        "runtime_spec": runtime,
        "runtime_contract": model_runtime_contract(model),
        "implementation_sha256": {
            str(path.relative_to(project_root)): sha256_path(path)
            for path in implementation_paths
        },
    }
    return _canonical_sha(payload)


def smoke_receipt_path(final_root, model_id):
    return Path(final_root) / "model_smokes" / model_id / "SMOKE_RECEIPT.json"


def _daemon_status(model_ids, complete, terminal):
    if len(complete) == len(model_ids):
        return "ok"
    if len(set(complete) | set(terminal)) == len(model_ids):
        return "failed"
    return "running"


def valid_smoke_receipt(final_root, model_id, contract_sha256=None):
    path = smoke_receipt_path(final_root, model_id)
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    canonical = dict(receipt); stored = canonical.pop("receipt_sha256", None)
    if (
        receipt.get("status") != "ok" or receipt.get("model_id") != model_id
        or receipt.get("protocol_id") != MODEL_SMOKE_PROTOCOL_ID
        or (contract_sha256 is not None and receipt.get("contract_sha256") != contract_sha256)
        or stored != _canonical_sha(canonical)
    ):
        return None
    for artifact in receipt.get("artifacts") or []:
        target = Path(artifact["path"])
        if not target.is_file() or sha256_path(target) != artifact["sha256"]:
            return None
    return receipt


def _archive_stale_smoke_generation(final_root, model_id, expected_contract_sha256):
    model_root = Path(final_root) / "model_smokes" / model_id
    receipt_path = model_root / "SMOKE_RECEIPT.json"
    try:
        old_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old_receipt = {}
    candidates = [
        model_root / "cache", model_root / "token_cache", receipt_path,
        model_root / "TERMINAL_FAILURE.json",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    old_contract = str(old_receipt.get("contract_sha256") or "unknown")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = model_root / "archive" / f"contract_{old_contract[:12]}_{stamp}"
    suffix = 1
    while archive.exists():
        archive = model_root / "archive" / f"contract_{old_contract[:12]}_{stamp}_{suffix:02d}"
        suffix += 1
    archive.mkdir(parents=True)
    moved = []
    for source in existing:
        target = archive / source.name
        os.replace(source, target)
        moved.append(source.name)
    _write_json(archive / "ARCHIVE_MANIFEST.json", {
        "status": "archived", "model_id": model_id,
        "reason": "smoke_contract_changed_or_active_artifact_invalid",
        "old_contract_sha256": old_contract,
        "expected_contract_sha256": expected_contract_sha256,
        "moved": moved,
        "archived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return archive


def terminal_smoke_failure_path(final_root, model_id):
    return Path(final_root) / "model_smokes" / model_id / "TERMINAL_FAILURE.json"


def valid_terminal_smoke_failure(final_root, model_id, contract_sha256=None):
    path = terminal_smoke_failure_path(final_root, model_id)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    canonical = dict(receipt); stored = canonical.pop("receipt_sha256", None)
    if (
        receipt.get("status") != "terminal_failed" or receipt.get("model_id") != model_id
        or receipt.get("protocol_id") != MODEL_SMOKE_PROTOCOL_ID
        or (contract_sha256 is not None and receipt.get("contract_sha256") != contract_sha256)
        or stored != _canonical_sha(canonical)
    ):
        return None
    for artifact in receipt.get("artifacts") or []:
        target = Path(artifact["path"])
        if not target.is_file() or sha256_path(target) != artifact["sha256"]:
            return None
    return receipt


def _smoke_dataset(final_root):
    root = Path(final_root) / "model_smokes" / "dataset" / "A01"
    if (root / "dataset_manifest.json").is_file():
        return root
    rows = []
    for split, count in (("train", 4), ("validation", 4), ("test", 4)):
        for index in range(count):
            digest = hashlib.sha256(f"{split}:{index}".encode()).digest()
            sequence = "".join("ACGT"[value % 4] for value in digest * 2)
            rows.append({
                "sample_id": f"{split}.{index}", "split": split, "sequence": sequence,
                "label": index % 2, "species": "synthetic_crop", "group_id": f"{split}.{index}",
            })
    root.parent.mkdir(parents=True, exist_ok=True)
    materialize_rows(rows, root, "A01", "binary_classification", {"source_id": "generated_model_smoke"})
    return root


def run_public_model_smoke(registry, model_id, project_root, final_root):
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    model = next(row for row in registry["models"] if row["model_id"] == model_id)
    contract_sha = smoke_contract_sha256(model, project_root)
    resumed = valid_smoke_receipt(final_root, model_id, contract_sha)
    if resumed:
        return {**resumed, "resumed": True}
    if model["kind"] != "public_weight":
        raise ValueError("model smoke is restricted to frozen public weights")
    _archive_stale_smoke_generation(final_root, model_id, contract_sha)
    dataset = _smoke_dataset(final_root)
    output = final_root / "model_smokes" / model_id / "cache"
    command = build_encode_command(
        registry, "A01", model_id, project_root, dataset, output,
        context=64, device="cuda",
    )
    extra_env = {}
    compiler_dir = Path(__import__("sys").executable).resolve().parent
    cc = compiler_dir / "x86_64-conda-linux-gnu-cc"
    cxx = compiler_dir / "x86_64-conda-linux-gnu-c++"
    if cc.is_file():
        extra_env["CC"] = str(cc)
    if cxx.is_file():
        extra_env["CXX"] = str(cxx)
    if model.get("runtime_pythonpath"):
        runtime = project_root / model["runtime_pythonpath"]
        inherited = os.environ.get("PYTHONPATH")
        extra_env["PYTHONPATH"] = str(runtime) + (os.pathsep + inherited if inherited else "")
    started = time.time()
    execution = run_with_dynamic_gpu(
        command, f"downstream-final:model-smoke:{model_id}", cwd=project_root,
        extra_env=extra_env, required_hostname="gpu05",
        allowed_gpu_indices=range(7),
    )
    if execution["returncode"]:
        raise RuntimeError(f"model smoke command exited {execution['returncode']}")
    cache = output / model_id / "context_64" / "A01.npz"
    manifest = output / model_id / "context_64" / "cache_manifest.json"
    with np.load(cache, allow_pickle=False) as payload:
        embeddings = payload["embeddings"]
        if embeddings.shape[0] != 12 or embeddings.ndim != 2 or embeddings.shape[1] < 1:
            raise RuntimeError(f"invalid smoke embedding shape: {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise RuntimeError("non-finite smoke embeddings")
    runtime = runtime_spec(model_id)
    executions = {"pooled": execution}
    artifacts = [
        {"path": str(path), "sha256": sha256_path(path), "size_bytes": path.stat().st_size}
        for path in (cache, manifest)
    ]
    if runtime.get("token_aligned"):
        token_dataset = final_root / "datasets" / "B13"
        if not (token_dataset / "samples.tsv").is_file():
            raise RuntimeError("token-aligned model smoke requires canonical B13 samples")
        token_output = final_root / "model_smokes" / model_id / "token_cache"
        token_command = build_encode_command(
            registry, "B13", model_id, project_root, token_dataset, token_output,
            context=64, device="cuda",
        )
        token_execution = run_with_dynamic_gpu(
            token_command, f"downstream-final:model-token-smoke:{model_id}",
            cwd=project_root, extra_env=extra_env, required_hostname="gpu05",
            allowed_gpu_indices=range(7),
        )
        if token_execution["returncode"]:
            raise RuntimeError(f"model token smoke command exited {token_execution['returncode']}")
        token_root = token_output / model_id / "context_64"
        token_manifest_path = token_root / "cache_manifest.json"
        token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
        hidden_shape = tuple(int(value) for value in token_manifest["hidden_shape"])
        labels_shape = tuple(int(value) for value in token_manifest["labels_shape"])
        if len(hidden_shape) != 3 or hidden_shape[:2] != labels_shape or hidden_shape[1] != 64:
            raise RuntimeError(f"invalid token smoke shapes: hidden={hidden_shape} labels={labels_shape}")
        hidden_path = Path(token_manifest["hidden_path"])
        labels_path = Path(token_manifest["labels_path"])
        selected_path = Path(token_manifest["selected_samples_path"])
        expected_hidden_bytes = int(np.prod(hidden_shape)) * np.dtype(np.float16).itemsize
        if hidden_path.stat().st_size != expected_hidden_bytes:
            raise RuntimeError("token smoke hidden memmap size mismatch")
        hidden = np.memmap(hidden_path, dtype=np.float16, mode="r", shape=hidden_shape)
        if not np.isfinite(hidden).all():
            raise RuntimeError("non-finite token smoke embeddings")
        if labels_path.stat().st_size != int(np.prod(labels_shape)):
            raise RuntimeError("token smoke labels memmap size mismatch")
        executions["token"] = token_execution
        artifacts.extend(
            {"path": str(path), "sha256": sha256_path(path), "size_bytes": path.stat().st_size}
            for path in (token_manifest_path, hidden_path, labels_path, selected_path)
        )
    receipt = {
        "status": "ok", "model_id": model_id, "revision": model["revision"],
        "protocol_id": MODEL_SMOKE_PROTOCOL_ID, "contract_sha256": contract_sha,
        "context_bp": 64, "runtime_spec": runtime,
        "runtime_contract": model_runtime_contract(model), "execution": execution,
        "executions": executions,
        "artifacts": artifacts, "elapsed_seconds": time.time() - started,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resumed": False,
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    _write_json(smoke_receipt_path(final_root, model_id), receipt)
    return receipt


def _write_terminal_smoke_failure(model, project_root, final_root, attempts):
    model_id = model["model_id"]
    contract_sha = smoke_contract_sha256(model, project_root)
    log_path = Path(final_root) / "model_smokes" / model_id / "worker.log"
    artifacts = []
    if log_path.is_file():
        artifacts.append({
            "path": str(log_path), "sha256": sha256_path(log_path),
            "size_bytes": log_path.stat().st_size,
        })
    receipt = {
        "status": "terminal_failed", "model_id": model_id,
        "revision": model["revision"], "protocol_id": MODEL_SMOKE_PROTOCOL_ID,
        "contract_sha256": contract_sha, "attempts": int(attempts),
        "runtime_spec": runtime_spec(model_id),
        "runtime_contract": model_runtime_contract(model), "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    _write_json(terminal_smoke_failure_path(final_root, model_id), receipt)
    return receipt


def run_smoke_daemon(registry, project_root, final_root, script_path,
                     max_workers=3, poll_seconds=60, max_attempts=3):
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    model_rows = {
        row["model_id"]: row for row in registry["models"] if row["kind"] == "public_weight"
    }
    model_ids = list(model_rows)
    contracts = {
        model_id: smoke_contract_sha256(model, project_root)
        for model_id, model in model_rows.items()
    }
    attempts_path = final_root / "MODEL_SMOKE_ATTEMPTS.json"
    try:
        attempt_state = json.loads(attempts_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        attempt_state = {"records": {}}
    attempts = defaultdict(int)
    for model_id in model_ids:
        record = attempt_state.get("records", {}).get(model_id, {})
        if record.get("contract_sha256") == contracts[model_id]:
            attempts[model_id] = int(record.get("attempts", 0))
    active = {}
    status_path = final_root / "MODEL_SMOKE_STATUS.json"
    while True:
        for model_id, item in list(active.items()):
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["handle"].close(); del active[model_id]
            if returncode:
                attempts[model_id] += 1
                attempt_state.setdefault("records", {})[model_id] = {
                    "contract_sha256": contracts[model_id],
                    "attempts": attempts[model_id],
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                _write_json(attempts_path, attempt_state)
        complete = {
            model_id for model_id in model_ids
            if valid_smoke_receipt(final_root, model_id, contracts[model_id])
        }
        for model_id in model_ids:
            if (
                model_id not in complete and attempts[model_id] >= max_attempts
                and not valid_terminal_smoke_failure(final_root, model_id, contracts[model_id])
            ):
                _write_terminal_smoke_failure(
                    model_rows[model_id], project_root, final_root, attempts[model_id],
                )
        terminal = {
            model_id for model_id in model_ids
            if model_id not in complete
            and valid_terminal_smoke_failure(final_root, model_id, contracts[model_id])
        }
        waiting = {
            model_id: "missing_download_receipt"
            for model_id in model_ids
            if model_id not in complete and model_id not in terminal and model_id not in active
            and not (
                project_root / "training_server_transfer/public_models/downstream_v4"
                / model_id / "download_receipt.json"
            ).is_file()
        }
        slots = max(0, int(max_workers) - len(active))
        if slots:
            for model_id in model_ids:
                if not slots:
                    break
                if model_id in complete or model_id in terminal or model_id in active:
                    continue
                download = (
                    project_root / "training_server_transfer/public_models/downstream_v4"
                    / model_id / "download_receipt.json"
                )
                if not download.is_file():
                    waiting[model_id] = "missing_download_receipt"
                    continue
                root = final_root / "model_smokes" / model_id; root.mkdir(parents=True, exist_ok=True)
                handle = (root / "worker.log").open("ab", buffering=0)
                command = [
                    str(Path.home() / ".local/share/mamba/envs/zuowu_genomemodel/bin/python"),
                    str(Path(script_path).resolve()), "run-model-smoke", "--model-id", model_id,
                    "--final-root", str(final_root),
                ]
                process = subprocess.Popen(
                    command, cwd=project_root, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[model_id] = {"process": process, "handle": handle, "pid": process.pid}
                slots -= 1; time.sleep(3)
        payload = {
            "status": _daemon_status(model_ids, complete, terminal),
            "models_total": len(model_ids), "models_complete": len(complete),
            "active": {key: value["pid"] for key, value in active.items()},
            "waiting": dict(sorted(waiting.items())),
            "terminal_failures": sorted(terminal),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _write_json(status_path, payload)
        if payload["status"] in {"ok", "failed"}:
            return payload
        time.sleep(int(poll_seconds))
