#!/usr/bin/env bash
# Generic race-to-run task runner (pattern from occams_multimodal).
# Env: TASK_NAME (required), TASK_CMD (required), CONDA_ENV (default pathology)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_NAME="${TASK_NAME:?TASK_NAME required}"
TASK_CMD="${TASK_CMD:?TASK_CMD required}"
CONDA_ENV="${CONDA_ENV:-pathology}"

TASK_DIR="${ROOT_DIR}/runs/${TASK_NAME}"
LOCK_DIR="${TASK_DIR}/.lock"
DONE_FILE="${TASK_DIR}/done.json"
LOG_DIR="${TASK_DIR}/logs"
mkdir -p "${TASK_DIR}" "${LOG_DIR}"

SELF_JOB_ID="${SLURM_JOB_ID:-no_slurm_jobid}"
SELF_PARTITION="${SLURM_JOB_PARTITION:-unknown}"

cleanup_lock() {
  if [[ -d "${LOCK_DIR}" ]] && [[ "$(cat "${LOCK_DIR}/job_id" 2>/dev/null)" == "${SELF_JOB_ID}" ]]; then
    rm -rf "${LOCK_DIR}" || true
  fi
}
trap cleanup_lock EXIT

if [[ -f "${DONE_FILE}" ]]; then echo "[skip] done: ${DONE_FILE}"; exit 0; fi

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  OTHER="$(cat "${LOCK_DIR}/job_id" 2>/dev/null || true)"
  # a preempted+requeued job keeps its SLURM_JOB_ID, so a lock holding our own id is ours: reclaim it (resume from checkpoint)
  if [[ "${OTHER}" == "${SELF_JOB_ID}" ]]; then
    echo "[reclaim] lock held by this job id (requeue, restart ${SLURM_RESTART_COUNT:-?})"
  elif [[ -n "${OTHER}" ]] && [[ -n "$(squeue -h -j "${OTHER}" 2>/dev/null)" ]]; then
    echo "[skip] owned by live job ${OTHER}"; exit 0
  fi
  rm -rf "${LOCK_DIR}" || true
  mkdir "${LOCK_DIR}" 2>/dev/null || { echo "[skip] lock retry lost"; exit 0; }
fi
echo "${SELF_JOB_ID}" > "${LOCK_DIR}/job_id"
echo "${SELF_PARTITION}" > "${LOCK_DIR}/partition"
date -Iseconds > "${LOCK_DIR}/started_at"
# lock acquired: cancel this task's still-PENDING twins on other partitions right away (never self; workers have other names)
for j in $(squeue -h -u "${USER}" -n "${TASK_NAME}" -t PENDING -o %i 2>/dev/null); do [[ "${j}" != "${SELF_JOB_ID}" ]] && scancel "${j}" 2>/dev/null || true; done

source /home/zuberi01/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

OUTDIR="${TASK_DIR}/output"
mkdir -p "${OUTDIR}"
export OUTDIR ROOT_DIR

set +e
bash -c "${TASK_CMD}" 2>&1 | tee "${LOG_DIR}/run_${SELF_JOB_ID}.log"
RC=${PIPESTATUS[0]}
set -e
if [[ ${RC} -ne 0 ]]; then
  echo "[fail] rc=${RC}; lock released, no done marker (another partition may retry)"
  exit "${RC}"
fi

cat > "${DONE_FILE}" <<EOF
{"task":"${TASK_NAME}","job_id":"${SELF_JOB_ID}","partition":"${SELF_PARTITION}","completed_at":"$(date -Iseconds)","output_dir":"${OUTDIR}"}
EOF
echo "[done] ${DONE_FILE}"
