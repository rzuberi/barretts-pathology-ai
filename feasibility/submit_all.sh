#!/usr/bin/env bash
# Submit the four thesis feasibility gates using the race-to-run pattern.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${ROOT_DIR}/slurm_logs"

submit() { # name, partitions(csv), extra sbatch args..., last arg = python task file
  local name="$1"; local parts="$2"; shift 2
  local extra=("$@"); local task="${extra[-1]}"; unset 'extra[-1]'
  IFS=',' read -r -a arr <<< "${parts}"
  for p in "${arr[@]}"; do
    local pextra=()
    [[ "$p" == "h200" ]] && pextra+=(--qos=h200_preempt)
    if ! sbatch --job-name="feas_${name}_${p}" --partition="$p" "${extra[@]}" "${pextra[@]}" \
      --output="${ROOT_DIR}/slurm_logs/${name}_${p}_%j.out" \
      --error="${ROOT_DIR}/slurm_logs/${name}_${p}_%j.err" \
      --export=ALL,TASK_NAME="${name}",TASK_CMD="python ${ROOT_DIR}/${task}",CONDA_ENV="${CONDA_ENV:-pathology}" \
      --wrap="bash ${ROOT_DIR}/run_task.sh"; then
      echo "[warn] submit failed: ${name} on ${p}" >&2
    fi
  done
}

# CPU gates -> epyc primary, cuda backup
submit occams_fusion   "epyc,cuda" --time=02:00:00 --cpus-per-task=8  --mem=32G task_occams_fusion.py
submit tcga_reports    "epyc,cuda" --time=02:00:00 --cpus-per-task=4  --mem=16G task_tcga_reports.py
submit tcga_esca_smoke "epyc,cuda" --time=03:00:00 --cpus-per-task=4  --mem=16G task_tcga_esca_smoke.py
# GPU gate -> cuda + h200 (CUDA-only pipeline; rocm excluded)
submit erin_uni2_smoke "cuda,h200" --time=03:00:00 --cpus-per-task=8 --mem=64G --gres=gpu:1 task_erin_uni2_smoke.py

echo "check: squeue -u \$USER -o '%.10i %.10P %.30j %.2t %.10M %R' | grep feas_"
