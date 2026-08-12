"""Revision-pinned dataset/model acquisition with machine receipts."""

import fnmatch
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import HfApi, snapshot_download

from .adapters import source_patterns
from .data import sha256_path
from .model_adapters import clean_model_runtime_cache


DEFAULT_DATA_ROOT = "training_server_transfer/external_data/downstream_v4"
DEFAULT_MODEL_ROOT = "training_server_transfer/public_models/downstream_v4"


def byte_ranges(total_bytes, parts=16):
    total_bytes = int(total_bytes); parts = max(1, int(parts))
    if total_bytes <= 0:
        return []
    parts = min(parts, total_bytes)
    quotient, remainder = divmod(total_bytes, parts)
    result = []
    start = 0
    for index in range(parts):
        width = quotient + (1 if index < remainder else 0)
        result.append((start, start + width - 1)); start += width
    return result


def _download_range(url, part_path, start, end, retries=12, read_timeout=45,
                    slow_window=120, min_bytes_per_second=32 * 1024):
    part_path = Path(part_path); expected = int(end) - int(start) + 1
    if part_path.is_file() and part_path.stat().st_size > expected:
        part_path.unlink()
    for attempt in range(1, int(retries) + 1):
        observed = part_path.stat().st_size if part_path.is_file() else 0
        if observed == expected:
            return observed
        request_start = int(start) + observed
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={request_start}-{int(end)}"},
        )
        try:
            attempt_started = time.monotonic(); attempt_bytes = 0
            with urllib.request.urlopen(request, timeout=read_timeout) as response:
                if response.status != 206:
                    raise RuntimeError(f"range server returned HTTP {response.status}")
                with part_path.open("ab") as handle:
                    while True:
                        reader = getattr(response, "read1", response.read)
                        chunk = reader(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk); attempt_bytes += len(chunk)
                        elapsed = time.monotonic() - attempt_started
                        if (
                            elapsed >= float(slow_window)
                            and attempt_bytes / elapsed < float(min_bytes_per_second)
                        ):
                            raise TimeoutError(
                                f"range progress below {min_bytes_per_second} B/s "
                                f"for {elapsed:.1f}s"
                            )
        except Exception as error:
            print(json.dumps({
                "status": "range_retry", "path": str(part_path),
                "attempt": attempt, "retries": int(retries),
                "bytes": part_path.stat().st_size if part_path.is_file() else 0,
                "error": repr(error),
            }), flush=True)
            if attempt == int(retries):
                raise
            time.sleep(min(30, 2 * attempt))
    raise RuntimeError(f"range download exhausted retries: {part_path}")


def download_hf_file_resumable(repo_id, revision, relative_path, destination,
                               expected_size, expected_sha256=None, workers=16,
                               repo_type="model"):
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(expected_size)
    if destination.is_file() and destination.stat().st_size == expected_size:
        observed = sha256_path(destination)
        if not expected_sha256 or observed == expected_sha256:
            return destination
    prefix = "datasets/" if repo_type == "dataset" else ""
    url = (
        f"https://huggingface.co/{prefix}{repo_id}/resolve/{revision}/"
        f"{quote(str(relative_path), safe='/')}"
    )
    part_root = destination.parent / f".{destination.name}.parts"
    part_root.mkdir(parents=True, exist_ok=True)
    ranges = byte_ranges(expected_size, workers)
    started = time.time()
    with ThreadPoolExecutor(max_workers=min(int(workers), len(ranges))) as executor:
        futures = {}
        for index, (start, end) in enumerate(ranges):
            part = part_root / f"{index:04d}.{start}-{end}.part"
            futures[executor.submit(_download_range, url, part, start, end)] = (index, part)
        completed = 0
        for future in as_completed(futures):
            future.result(); completed += 1
            downloaded = sum(path.stat().st_size for path in part_root.glob("*.part"))
            elapsed = max(time.time() - started, 1e-6)
            speed = downloaded / elapsed
            eta = (expected_size - downloaded) / speed if speed > 0 else None
            print(json.dumps({
                "status": "downloading_model_weight", "repo_id": repo_id,
                "path": str(relative_path), "parts_complete": completed,
                "parts_total": len(ranges), "bytes": downloaded,
                "total_bytes": expected_size, "bytes_per_second": speed,
                "eta_seconds": eta,
            }, allow_nan=False), flush=True)
    assembling = destination.with_name(f".{destination.name}.assembling")
    digest = hashlib.sha256(); written = 0
    with assembling.open("wb") as output:
        for index, (start, end) in enumerate(ranges):
            part = part_root / f"{index:04d}.{start}-{end}.part"
            expected_part = end - start + 1
            if not part.is_file() or part.stat().st_size != expected_part:
                raise RuntimeError(f"model weight part is incomplete: {part}")
            with part.open("rb") as handle:
                while True:
                    chunk = handle.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk); digest.update(chunk); written += len(chunk)
    observed_sha256 = digest.hexdigest()
    if written != expected_size:
        assembling.unlink(missing_ok=True)
        raise RuntimeError(f"assembled model weight size mismatch: {written} != {expected_size}")
    if expected_sha256 and observed_sha256 != expected_sha256:
        assembling.unlink(missing_ok=True)
        raise RuntimeError(
            f"assembled model weight SHA256 mismatch: {observed_sha256} != {expected_sha256}"
        )
    os.replace(assembling, destination)
    for part in part_root.glob("*.part"):
        part.unlink()
    part_root.rmdir()
    return destination


def select_repository_files(files, patterns):
    selected = {}
    for row in files:
        path = str(row["path"])
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            selected[path] = dict(row)
    if not selected:
        raise ValueError(f"repository patterns selected no files: {patterns}")
    return selected


def _repository_files(repo_id, revision, repo_type):
    api = HfApi()
    if repo_type == "model":
        info = api.model_info(repo_id, revision=revision, files_metadata=True, timeout=120)
    elif repo_type == "dataset":
        info = api.dataset_info(repo_id, revision=revision, files_metadata=True, timeout=120)
    else:
        raise ValueError(f"unsupported repository type: {repo_type}")
    records = []
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        records.append({
            "path": sibling.rfilename,
            "size_bytes": int(getattr(sibling, "size", 0) or 0),
            "lfs_sha256": getattr(lfs, "sha256", None) if lfs is not None else None,
        })
    return records


def _download_repository_selection(repo_id, revision, repo_type, destination,
                                   selected, max_workers):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    direct = {
        path: row for path, row in selected.items()
        if row.get("lfs_sha256") or int(row.get("size_bytes") or 0) > 8 * 1024 * 1024
    }
    metadata = sorted(set(selected) - set(direct))
    if metadata:
        snapshot_download(
            repo_id=repo_id, repo_type=repo_type, revision=revision,
            local_dir=str(destination), allow_patterns=metadata,
            max_workers=int(max_workers),
        )
    for relative, expected in sorted(direct.items()):
        download_hf_file_resumable(
            repo_id, revision, relative, destination / relative,
            expected["size_bytes"], expected.get("lfs_sha256"),
            workers=max(8, int(max_workers) * 2), repo_type=repo_type,
        )
    artifacts = []
    for relative, expected in sorted(selected.items()):
        path = destination / relative
        if not path.is_file():
            raise RuntimeError(f"selected repository file is missing: {path}")
        size = path.stat().st_size; digest = sha256_path(path)
        if expected.get("size_bytes") and size != int(expected["size_bytes"]):
            raise RuntimeError(f"repository file size mismatch: {relative}")
        if expected.get("lfs_sha256") and digest != expected["lfs_sha256"]:
            raise RuntimeError(f"repository file SHA256 mismatch: {relative}")
        artifacts.append({
            "path": relative, "size_bytes": size, "sha256": digest,
            "expected_lfs_sha256": expected.get("lfs_sha256"),
        })
    return artifacts


def select_model_snapshot_files(files, metadata_limit_bytes=64 * 1024 * 1024):
    """Select one runnable PyTorch weight format plus small runtime metadata."""
    records = {str(row["path"]): dict(row) for row in files}
    paths = sorted(records)
    safetensors = [
        path for path in paths
        if path.endswith(".safetensors") or path.endswith(".safetensors.index.json")
    ]
    pytorch_bins = [
        path for path in paths
        if Path(path).name.startswith("pytorch_model")
        and (path.endswith(".bin") or path.endswith(".bin.index.json"))
    ]
    standalone_pt = [path for path in paths if path.endswith((".pt", ".pth"))]
    weight_files = safetensors or pytorch_bins or standalone_pt
    if not weight_files:
        raise ValueError("model repository has no supported PyTorch weight file")

    def is_any_weight(path):
        return (
            path.endswith((".safetensors", ".safetensors.index.json", ".bin", ".bin.index.json", ".pt", ".pth", ".msgpack", ".h5"))
            or Path(path).name.startswith(("flax_model", "tf_model"))
        )

    metadata = []
    for path in paths:
        if path in weight_files:
            continue
        if path.startswith(("jax_model/", "flax_model/", "tf_model/")) or is_any_weight(path):
            continue
        if int(records[path].get("size_bytes") or 0) <= int(metadata_limit_bytes):
            metadata.append(path)
    allow_patterns = sorted(set(metadata + weight_files))
    if "config.json" not in allow_patterns:
        raise ValueError("model repository selection lacks config.json")
    expected = {
        path: {
            "size_bytes": int(records[path].get("size_bytes") or 0),
            "lfs_sha256": records[path].get("lfs_sha256"),
        }
        for path in allow_patterns
    }
    return {
        "allow_patterns": allow_patterns,
        "weight_files": sorted(weight_files),
        "expected_files": expected,
    }


def _model_repository_files(model_ref, revision):
    return _repository_files(model_ref, revision, "model")


def _repo_id(url, marker):
    if marker not in url:
        raise ValueError(f"cannot parse repository ID from {url!r}")
    return url.split(marker, 1)[1].strip("/")


def _artifact_receipt(root):
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.parts or path.name.endswith("receipt.json"):
            continue
        rows.append({
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        })
    return rows


def merge_source_download_scope(previous, source, selected_task_ids, profile, patterns):
    """Merge per-task download coverage without letting smoke replace formal data."""
    previous = previous if isinstance(previous, dict) else {}
    compatible = (
        previous.get("status") == "ok"
        and previous.get("source_id") == source["source_id"]
        and previous.get("revision") == source.get("revision")
    )
    task_profiles = {}
    previous_patterns = []
    if compatible:
        previous_patterns = list(previous.get("allow_patterns") or [])
        task_profiles.update(previous.get("task_profiles") or {})
        if not task_profiles:
            old_profile = previous.get("profile")
            for task_id in previous.get("selected_tasks") or []:
                task_profiles[task_id] = old_profile
    priority = {"smoke": 0, "formal": 1}
    for task_id in selected_task_ids:
        old = task_profiles.get(task_id)
        if old not in priority or priority[profile] >= priority[old]:
            task_profiles[task_id] = profile
    profiles = set(task_profiles.values())
    aggregate_profile = next(iter(profiles)) if len(profiles) == 1 else "mixed"
    return {
        "profile": aggregate_profile,
        "selected_tasks": sorted(task_profiles),
        "task_profiles": dict(sorted(task_profiles.items())),
        "allow_patterns": sorted(set(previous_patterns) | set(patterns)),
    }


def source_receipt_covers(receipt, source, task_ids, profile):
    """Require every requested task to have an equal-or-stronger data profile."""
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "ok"
        or receipt.get("source_id") != source["source_id"]
        or receipt.get("revision") != source.get("revision")
    ):
        return False
    priority = {"smoke": 0, "formal": 1}
    if profile not in priority:
        raise ValueError(f"unsupported source receipt profile: {profile}")
    selected = set(receipt.get("selected_tasks") or [])
    task_profiles = receipt.get("task_profiles") or {}
    for task_id in task_ids or []:
        if task_id not in selected:
            return False
        observed = task_profiles.get(task_id, receipt.get("profile"))
        if observed not in priority or priority[observed] < priority[profile]:
            return False
    return True


def _download_direct_file(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    filename = source["url"].rsplit("/", 1)[-1]
    target = destination / filename
    expected = str(source.get("revision", ""))
    if not expected.startswith("sha256:"):
        raise ValueError("direct_file source revision must be sha256:<digest>")
    expected_digest = expected.split(":", 1)[1]
    if target.is_file() and sha256_path(target) == expected_digest:
        return target
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=str(destination))
    os.close(descriptor)
    temporary = Path(temporary_name)
    downloaded = 0
    try:
        with urllib.request.urlopen(source["url"], timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded % (64 * 1024 * 1024) < len(chunk):
                    print(json.dumps({"status": "downloading", "source_id": source["source_id"], "bytes": downloaded}), flush=True)
        observed = sha256_path(temporary)
        if observed != expected_digest:
            raise RuntimeError(f"direct download SHA256 mismatch: {observed} != {expected_digest}")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def download_source(registry, source_id, project_root, task_ids=None, profile="formal", max_workers=8):
    project_root = Path(project_root).resolve()
    sources = {row["source_id"]: row for row in registry["sources"]}
    if source_id not in sources:
        raise ValueError(f"unknown source_id: {source_id}")
    source = sources[source_id]
    destination = project_root / DEFAULT_DATA_ROOT / source_id
    selected = [row for row in registry["tasks"] if row["source_id"] == source_id]
    if task_ids is not None:
        requested = set(task_ids)
        selected = [row for row in selected if row["task_id"] in requested]
        missing = requested - {row["task_id"] for row in selected}
        if missing:
            raise ValueError(f"tasks do not belong to source {source_id}: {sorted(missing)}")
    if source["kind"] == "local_artifact":
        local = project_root / source["url"].removeprefix("file://")
        if not local.exists():
            raise FileNotFoundError(local)
        return {"status": "ok", "source_id": source_id, "local_artifact": str(local.resolve()), "downloaded": False}
    destination.mkdir(parents=True, exist_ok=True)
    patterns = sorted({pattern for row in selected for pattern in source_patterns(row["task_id"], profile)})
    if source["kind"] == "huggingface_dataset":
        if not patterns:
            raise ValueError(f"no task-specific allow patterns for source {source_id}")
        repo_id = _repo_id(source["url"], "/datasets/")
        repository_files = _repository_files(repo_id, source["revision"], "dataset")
        selected_files = select_repository_files(
            repository_files, patterns + ["README.md", ".gitattributes"],
        )
        artifacts = _download_repository_selection(
            repo_id, source["revision"], "dataset", destination,
            selected_files, max_workers,
        )
    elif source["kind"] == "direct_file":
        _download_direct_file(source, destination)
        artifacts = _artifact_receipt(destination)
    else:
        raise ValueError(f"unsupported source kind: {source['kind']}")
    receipt_path = destination / "download_receipt.json"
    try:
        previous_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous_receipt = None
    scope = merge_source_download_scope(
        previous_receipt, source, [row["task_id"] for row in selected], profile, patterns,
    )
    receipt = {
        "status": "ok",
        "source_id": source_id,
        "source_kind": source["kind"],
        "source_url": source["url"],
        "revision": source.get("revision"),
        "license": source["license"],
        "license_state": source["license_state"],
        **scope,
        "files": len(artifacts),
        "total_bytes": sum(row["size_bytes"] for row in artifacts),
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (destination / "download_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return receipt


def download_model(registry, model_id, project_root, max_workers=8):
    project_root = Path(project_root).resolve()
    models = {row["model_id"]: row for row in registry["models"]}
    if model_id not in models:
        raise ValueError(f"unknown model_id: {model_id}")
    model = models[model_id]
    if model["kind"] != "public_weight":
        raise ValueError(f"model is not a downloadable public weight: {model_id}")
    destination = project_root / DEFAULT_MODEL_ROOT / model_id
    destination.mkdir(parents=True, exist_ok=True)
    selection = select_model_snapshot_files(
        _model_repository_files(model["model_ref"], model["revision"]),
    )
    metadata_patterns = [
        path for path in selection["allow_patterns"]
        if path not in set(selection["weight_files"])
    ]
    snapshot_download(
        repo_id=model["model_ref"],
        repo_type="model",
        revision=model["revision"],
        local_dir=str(destination),
        allow_patterns=metadata_patterns,
        max_workers=int(max_workers),
    )
    for relative in selection["weight_files"]:
        expected = selection["expected_files"][relative]
        download_hf_file_resumable(
            model["model_ref"], model["revision"], relative,
            destination / relative, expected["size_bytes"], expected["lfs_sha256"],
            workers=max(8, int(max_workers) * 2),
        )
    clean_model_runtime_cache(destination)
    artifacts = []
    for relative in selection["allow_patterns"]:
        path = destination / relative
        expected = selection["expected_files"][relative]
        if not path.is_file():
            raise RuntimeError(f"selected model file is missing after download: {path}")
        observed_size = path.stat().st_size
        observed_sha256 = sha256_path(path)
        if expected["size_bytes"] and observed_size != expected["size_bytes"]:
            raise RuntimeError(
                f"selected model file size mismatch: {relative}: "
                f"{observed_size} != {expected['size_bytes']}"
            )
        if expected["lfs_sha256"] and observed_sha256 != expected["lfs_sha256"]:
            raise RuntimeError(
                f"selected model file SHA256 mismatch: {relative}: "
                f"{observed_sha256} != {expected['lfs_sha256']}"
            )
        artifacts.append({
            "path": relative, "size_bytes": observed_size, "sha256": observed_sha256,
            "expected_lfs_sha256": expected["lfs_sha256"],
        })
    receipt = {
        "status": "ok",
        "model_id": model_id,
        "model_ref": model["model_ref"],
        "revision": model["revision"],
        "license": model["license"],
        "license_state": model["license_state"],
        "capabilities": model["capabilities"],
        "selected_files": selection["allow_patterns"],
        "selected_weight_files": selection["weight_files"],
        "files": len(artifacts),
        "total_bytes": sum(row["size_bytes"] for row in artifacts),
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (destination / "download_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return receipt
