"""Validate the downstream v4 Python/runtime lock."""

import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .data import sha256_path


IMPORT_PROBES = {
    "mamba-ssm": "mamba_ssm",
    "causal-conv1d": "causal_conv1d",
    "gpn": "gpn",
}
ENVIRONMENT_PROTOCOL_ID = "downstream_v4_runtime_lock_attestation_v1"


def _canonical_sha(payload):
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_environment(lock_path):
    lock_path = Path(lock_path).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    errors = []
    observed_packages = {}
    python_version = ".".join(map(str, sys.version_info[:3]))
    if python_version != lock["python"]:
        errors.append(f"python {python_version} != {lock['python']}")
    if str(Path(sys.executable).resolve()) != str(Path(lock["executable"]).resolve()):
        errors.append(f"executable {sys.executable} != {lock['executable']}")
    for package, expected in lock["packages"].items():
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            observed = None
        observed_packages[package] = observed
        if observed != expected:
            errors.append(f"{package} {observed} != {expected}")
    import_errors = {}
    for package, module in IMPORT_PROBES.items():
        try:
            importlib.import_module(module)
            import_errors[package] = None
        except Exception as error:
            import_errors[package] = f"{type(error).__name__}: {error}"
            errors.append(f"{package} import failed: {error}")
    import torch
    torch_version = torch.__version__
    cuda_runtime = torch.version.cuda
    cxx11_abi = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    if torch_version != lock["torch"]:
        errors.append(f"torch {torch_version} != {lock['torch']}")
    if cuda_runtime != lock["cuda_runtime"]:
        errors.append(f"torch CUDA {cuda_runtime} != {lock['cuda_runtime']}")
    if cxx11_abi != bool(lock["torch_cxx11_abi"]):
        errors.append(f"torch CXX11 ABI {cxx11_abi} != {lock['torch_cxx11_abi']}")
    gpn_direct_url = None
    try:
        gpn_direct_url = json.loads(importlib.metadata.distribution("gpn").read_text("direct_url.json") or "null")
    except Exception as error:
        errors.append(f"cannot read GPN direct_url.json: {error}")
    expected_commit = lock["git_dependencies"]["gpn"]["commit"]
    observed_commit = ((gpn_direct_url or {}).get("vcs_info") or {}).get("commit_id")
    if observed_commit != expected_commit:
        errors.append(f"gpn commit {observed_commit} != {expected_commit}")
    return {
        "status": "ok" if not errors else "failed",
        "lock_path": str(lock_path), "lock_sha256": sha256_path(lock_path),
        "python": python_version, "executable": str(Path(sys.executable).resolve()),
        "packages": observed_packages, "import_errors": import_errors,
        "torch": torch_version, "torch_cuda": cuda_runtime,
        "torch_cxx11_abi": cxx11_abi, "cuda_available": torch.cuda.is_available(),
        "gpn_commit": observed_commit, "errors": errors,
    }


def write_environment_receipt(lock_path, output_path, plan_sha256):
    report = validate_environment(lock_path)
    receipt = {
        "status": report["status"], "protocol_id": ENVIRONMENT_PROTOCOL_ID,
        "plan_sha256": str(plan_sha256), "validation": report,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    _atomic_json(output_path, receipt)
    return receipt


def valid_environment_receipt(path, lock_path, plan_sha256):
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    canonical = dict(receipt); stored = canonical.pop("receipt_sha256", None)
    if (
        receipt.get("status") != "ok"
        or receipt.get("protocol_id") != ENVIRONMENT_PROTOCOL_ID
        or receipt.get("plan_sha256") != str(plan_sha256)
        or stored != _canonical_sha(canonical)
    ):
        return None
    validation = receipt.get("validation") or {}
    if (
        validation.get("status") != "ok"
        or validation.get("lock_sha256") != sha256_path(lock_path)
        or validation.get("errors") != []
    ):
        return None
    return receipt
