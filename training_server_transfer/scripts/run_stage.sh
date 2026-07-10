#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?usage: bash scripts/run_stage.sh Stage_B <num_gpus>}"
NUM_GPUS="${2:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-zuowu_genomemodel}"
RESUME_CHECKPOINT=""
RESUME_SHA256=""
RESUME_MODE=""
MODEL_CONFIG="configs/model_large.json"
RUN_DIR=""
TRAIN_PID=""

forward_stop() {
  local signal_name="${1}"
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -s "${signal_name}" "${TRAIN_PID}"
    wait "${TRAIN_PID}" || true
  fi
}

trap 'forward_stop TERM' TERM
trap 'forward_stop INT' INT

if [[ -n "${PYTHON_BIN_OVERRIDE:-}" ]]; then
  [[ -x "${PYTHON_BIN_OVERRIDE}" ]] || { echo "PYTHON_BIN_OVERRIDE is not executable: ${PYTHON_BIN_OVERRIDE}" >&2; exit 7; }
  PYTHON_CMD=("${PYTHON_BIN_OVERRIDE}")
  TORCHRUN_CMD=("${PYTHON_BIN_OVERRIDE}" -m torch.distributed.run)
elif [[ -n "${CONDA_ENV_PREFIX:-}" ]]; then
  [[ -x "${CONDA_ENV_PREFIX}/bin/python" ]] || { echo "CONDA_ENV_PREFIX has no executable bin/python: ${CONDA_ENV_PREFIX}" >&2; exit 7; }
  PYTHON_CMD=("${CONDA_ENV_PREFIX}/bin/python")
  TORCHRUN_CMD=("${CONDA_ENV_PREFIX}/bin/python" -m torch.distributed.run)
elif command -v mamba >/dev/null 2>&1; then
  PYTHON_CMD=(mamba run -n "${CONDA_ENV}" python)
  TORCHRUN_CMD=(mamba run -n "${CONDA_ENV}" torchrun)
elif [[ -x "${HOME}/.local/share/mamba/envs/${CONDA_ENV}/bin/python" ]]; then
  PYTHON_CMD=("${HOME}/.local/share/mamba/envs/${CONDA_ENV}/bin/python")
  TORCHRUN_CMD=("${HOME}/.local/share/mamba/envs/${CONDA_ENV}/bin/python" -m torch.distributed.run)
else
  PYTHON_CMD=(python)
  TORCHRUN_CMD=(torchrun)
fi

case "${STAGE}" in
  Stage_B) CONFIG="configs/train_stage_B.json" ;;
  Stage_C1)
    CONFIG="configs/train_stage_C1.json"
    MODEL_CONFIG="configs/model_stage_C1_64k.json"
    RUN_DIR="runs/Stage_C1"
    RESUME_MODE="${RUN_MODE:-warmstart}"
    if [[ "${RESUME_MODE}" == "warmstart" ]]; then
      RESUME_CHECKPOINT="runs/Stage_B_cropgenome_fm_v2_stable/checkpoints/checkpoint_stage_B_8k_final.pt"
      RESUME_SHA256="c81bce39ec448845e929e755530bc7023a345cca42234ff7fb776f5f39c83fed"
    elif [[ "${RESUME_MODE}" == "exact" ]]; then
      RESUME_CHECKPOINT="${RESUME_CHECKPOINT_OVERRIDE:?RUN_MODE=exact requires RESUME_CHECKPOINT_OVERRIDE}"
      RESUME_SHA256="${RESUME_SHA256_OVERRIDE:-}"
    else
      echo "Unsupported Stage_C1 RUN_MODE: ${RESUME_MODE}" >&2
      exit 5
    fi
    ;;
  Stage_C2) CONFIG="configs/train_stage_C2.json" ;;
  Stage_D) CONFIG="configs/train_stage_D.json" ;;
  *) echo "Unknown stage: ${STAGE}" >&2; exit 2 ;;
esac

cd "${ROOT}"
if [[ -n "${RUN_DIR}" ]]; then
  mkdir -p "${RUN_DIR}"
  PID_FILE="${RUN_DIR}/launcher.pid"
  LOCK_DIR="${RUN_DIR}/.launcher.lock"
  if [[ -f "${PID_FILE}" ]]; then
    read -r EXISTING_PID < "${PID_FILE}" || true
    if [[ "${EXISTING_PID:-}" =~ ^[0-9]+$ ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
      echo "${STAGE} is already running with PID ${EXISTING_PID}" >&2
      exit 6
    fi
  fi
  rmdir "${LOCK_DIR}" 2>/dev/null || true
  if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "${STAGE} is already running (launcher lock exists)" >&2
    exit 6
  fi
  trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT
  printf '%s\n' "$$" > "${PID_FILE}.tmp"
  mv "${PID_FILE}.tmp" "${PID_FILE}"
fi
"${PYTHON_CMD[@]}" scripts/check_package.py --stage "${STAGE}" --quick

TRAIN_ARGS=(
  --data-root .
  --config "${CONFIG}"
  --model-config "${MODEL_CONFIG}"
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  [[ -f "${RESUME_CHECKPOINT}" ]] || { echo "Missing resume checkpoint: ${RESUME_CHECKPOINT}" >&2; exit 3; }
  if [[ -n "${RESUME_SHA256}" ]]; then
    read -r ACTUAL_SHA256 _ <<< "$(sha256sum "${RESUME_CHECKPOINT}")"
    [[ "${ACTUAL_SHA256}" == "${RESUME_SHA256}" ]] || {
      echo "Locked checkpoint SHA256 mismatch: expected=${RESUME_SHA256} actual=${ACTUAL_SHA256}" >&2
      exit 4
    }
  fi
  TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
  if [[ -n "${RESUME_MODE}" ]]; then
    TRAIN_ARGS+=(--resume-mode "${RESUME_MODE}")
  fi
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--dry-run)
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node="${NUM_GPUS}" \
    scripts/train.py "${TRAIN_ARGS[@]}" &
else
  "${PYTHON_CMD[@]}" scripts/train.py "${TRAIN_ARGS[@]}" &
fi
TRAIN_PID=$!
wait "${TRAIN_PID}"
