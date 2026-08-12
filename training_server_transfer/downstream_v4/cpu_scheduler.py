"""Submit ready final downstream CPU probe rows as resumable SLURM waves."""

import fcntl
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .final_controller import (
    _valid_cpu_row_receipt, _valid_terminal_cpu_failure, cpu_row_readiness,
)
from .sensitivity import sensitivity_readiness, valid_sensitivity_receipt


CPU_PARTITION = "q05"
CPU_EXECUTION_HOST = "gpu05"
SLURM_CPU_SUBMISSION_DISABLED = (
    "SLURM CPU submission is disabled by user policy; "
    "run CPU-only downstream work on gpu05 local CPUs"
)


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _active_job_ids():
    completed = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%A"],
        check=True, text=True, capture_output=True, timeout=60,
    )
    return {line.strip() for line in completed.stdout.splitlines() if line.strip().isdigit()}


def _is_cpu_row(row):
    if row.get("execution_kind") != "evaluation":
        return False
    task_kind = row["task_kind"]
    return task_kind != "token_multiclass" and not task_kind.startswith("zero_shot")


def ready_cpu_rows(plan, final_root, history, max_attempts=3):
    final_root = Path(final_root)
    active_ids = _active_job_ids()
    active_keys = set()
    for wave in history.get("waves", []):
        if wave.get("plan_sha256") != plan["plan_sha256"]:
            continue
        keys = wave.get("run_keys", [])
        if str(wave.get("job_id")) in active_ids:
            active_keys.update(keys)
    ready = []
    for row in plan["rows"]:
        key = row["run_key"]
        if not _is_cpu_row(row) or key in active_keys:
            continue
        if _valid_cpu_row_receipt(row, final_root, plan["plan_sha256"]):
            continue
        if _valid_terminal_cpu_failure(row, final_root, plan["plan_sha256"]):
            continue
        if cpu_row_readiness(row, final_root)[0]:
            ready.append(row)
    return ready


def _write_wave_files(project_root, final_root, chunks, wave_id, plan_sha256):
    raise RuntimeError(SLURM_CPU_SUBMISSION_DISABLED)
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    wave_root = final_root / "cpu_waves" / wave_id
    logs = final_root / "cpu_logs"
    wave_root.mkdir(parents=True, exist_ok=False); logs.mkdir(parents=True, exist_ok=True)
    manifest = wave_root / "WAVE.json"
    _atomic_json(manifest, {
        "wave_id": wave_id, "plan_sha256": plan_sha256, "chunks": chunks,
        "partition": CPU_PARTITION,
    })
    sbatch_path = wave_root / "run.sbatch"
    python = project_root.parent.parent / ".local/share/mamba/envs/zuowu_genomemodel/bin/python"
    # project_root is under ~/myhermes; derive the user's home without shell expansion.
    python = Path.home() / ".local/share/mamba/envs/zuowu_genomemodel/bin/python"
    script = project_root / "scripts/run_cropgenome_downstream_final.py"
    env_lib = Path.home() / ".local/share/mamba/envs/zuowu_genomemodel/lib"
    content = f"""#!/bin/sh
#SBATCH -J cgfm_final_probe
#SBATCH -p {CPU_PARTITION}
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --array=0-{len(chunks) - 1}%32
#SBATCH -o {logs}/probe-%A_%a.out
#SBATCH -e {logs}/probe-%A_%a.err

set -eu
if [ "${{SLURM_JOB_PARTITION:-}}" != "{CPU_PARTITION}" ]; then
    printf '%s\n' "ERROR: CPU downstream jobs are restricted to {CPU_PARTITION}; got ${{SLURM_JOB_PARTITION:-unset}}" >&2
    exit 64
fi
export PATH={python.parent}:$PATH
export LD_LIBRARY_PATH={env_lib}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd {project_root}
{python} -u {script} --project-root {project_root} cpu-wave --manifest {manifest} --final-root {final_root}
"""
    sbatch_path.write_text(content, encoding="utf-8")
    sbatch_path.chmod(0o755)
    return manifest, sbatch_path


def _write_sensitivity_sbatch(project_root, final_root, max_rows=32):
    raise RuntimeError(SLURM_CPU_SUBMISSION_DISABLED)
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    if int(max_rows) < 1:
        raise ValueError("max_rows must be positive")
    control_root = final_root / "controller" / "sensitivity"
    logs = final_root / "cpu_logs"
    control_root.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    sbatch_path = control_root / "run.sbatch"
    python = Path.home() / ".local/share/mamba/envs/zuowu_genomemodel/bin/python"
    script = project_root / "scripts/run_cropgenome_downstream_final.py"
    env_lib = Path.home() / ".local/share/mamba/envs/zuowu_genomemodel/lib"
    content = f"""#!/bin/sh
#SBATCH -J cgfm_final_sensitivity
#SBATCH -p {CPU_PARTITION}
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o {logs}/sensitivity-%j.out
#SBATCH -e {logs}/sensitivity-%j.err

set -eu
if [ "${{SLURM_JOB_PARTITION:-}}" != "{CPU_PARTITION}" ]; then
    printf '%s\n' "ERROR: CPU downstream jobs are restricted to {CPU_PARTITION}; got ${{SLURM_JOB_PARTITION:-unset}}" >&2
    exit 64
fi
export PATH={python.parent}:$PATH
export LD_LIBRARY_PATH={env_lib}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd {project_root}
{python} -u {script} --project-root {project_root} run-ready-sensitivity --max-rows {int(max_rows)} --final-root {final_root}
"""
    sbatch_path.write_text(content, encoding="utf-8")
    sbatch_path.chmod(0o755)
    return sbatch_path


def submit_ready_wave(project_root, final_root, plan, max_rows=256, chunk_size=4, max_attempts=3):
    raise RuntimeError(SLURM_CPU_SUBMISSION_DISABLED)
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    controller = final_root / "controller"
    controller.mkdir(parents=True, exist_ok=True)
    lock_path = controller / "cpu_submit.lock"
    history_path = controller / "CPU_SUBMISSIONS.json"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else {"waves": []}
        rows = ready_cpu_rows(plan, final_root, history, max_attempts=max_attempts)[:max_rows]
        if not rows:
            return {"status": "waiting", "submitted": 0}
        keys = [row["run_key"] for row in rows]
        chunks = [keys[index:index + chunk_size] for index in range(0, len(keys), chunk_size)]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:10]
        wave_id = f"{stamp}_{digest}"
        manifest, sbatch_path = _write_wave_files(
            project_root, final_root, chunks, wave_id, plan["plan_sha256"],
        )
        syntax = subprocess.run(["sh", "-n", str(sbatch_path)], text=True, capture_output=True, timeout=60)
        if syntax.returncode:
            raise RuntimeError(syntax.stderr)
        submitted = subprocess.run(
            ["sbatch", str(sbatch_path)], check=True, text=True, capture_output=True, timeout=60,
        )
        match = re.search(r"Submitted batch job (\d+)", submitted.stdout)
        if not match:
            raise RuntimeError(f"unexpected sbatch output: {submitted.stdout!r}")
        job_id = match.group(1)
        wave = {
            "wave_id": wave_id, "job_id": job_id, "manifest": str(manifest),
            "sbatch": str(sbatch_path), "run_keys": keys, "chunks": len(chunks),
            "plan_sha256": plan["plan_sha256"], "partition": CPU_PARTITION,
            "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        history.setdefault("waves", []).append(wave)
        history["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _atomic_json(history_path, history)
        return {"status": "submitted", "submitted": len(keys), "job_id": job_id, "chunks": len(chunks)}


def submit_ready_sensitivity(project_root, final_root, plan, registry, max_rows=32):
    raise RuntimeError(SLURM_CPU_SUBMISSION_DISABLED)
    project_root = Path(project_root).resolve(); final_root = Path(final_root).resolve()
    controller = final_root / "controller"
    controller.mkdir(parents=True, exist_ok=True)
    lock_path = controller / "sensitivity_submit.lock"
    history_path = controller / "SENSITIVITY_SUBMISSIONS.json"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.is_file() else {"submissions": []}
        )
        active_ids = _active_job_ids()
        active = [
            item for item in history.get("submissions", [])
            if str(item.get("job_id")) in active_ids
        ]
        if active:
            return {"status": "waiting", "submitted": 0, "active_job_ids": [item["job_id"] for item in active]}
        ready = []
        for row in plan["rows"]:
            if row.get("task_id") != "B17":
                continue
            if valid_sensitivity_receipt(row, final_root, plan["plan_sha256"]):
                continue
            if _valid_terminal_cpu_failure(row, final_root, plan["plan_sha256"]):
                continue
            if sensitivity_readiness(row, registry, final_root)[0]:
                ready.append(row)
        if not ready:
            return {"status": "waiting", "submitted": 0}
        selected = ready[:int(max_rows)]
        sbatch_path = _write_sensitivity_sbatch(
            project_root, final_root, max_rows=len(selected),
        )
        syntax = subprocess.run(
            ["sh", "-n", str(sbatch_path)], text=True,
            capture_output=True, timeout=60,
        )
        if syntax.returncode:
            raise RuntimeError(syntax.stderr)
        submitted = subprocess.run(
            ["sbatch", str(sbatch_path)], check=True, text=True,
            capture_output=True, timeout=60,
        )
        match = re.search(r"Submitted batch job (\d+)", submitted.stdout)
        if not match:
            raise RuntimeError(f"unexpected sbatch output: {submitted.stdout!r}")
        job_id = match.group(1)
        record = {
            "job_id": job_id, "sbatch": str(sbatch_path),
            "run_keys": [row["run_key"] for row in selected],
            "partition": CPU_PARTITION,
            "plan_sha256": plan["plan_sha256"],
            "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        history.setdefault("submissions", []).append(record)
        history["updated_at"] = record["submitted_at"]
        _atomic_json(history_path, history)
        return {
            "status": "submitted", "submitted": len(selected),
            "job_id": job_id, "partition": CPU_PARTITION,
        }
