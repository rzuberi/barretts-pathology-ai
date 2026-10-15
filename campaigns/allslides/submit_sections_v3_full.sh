#!/usr/bin/env bash
# Full-corpus per-section jury v3: one race-to-run task per (juror, shard), twinned on
# h200 (+qos) and cuda. Resume-capable worker, so preemption/timeouts just re-queue.
# Usage: submit_sections_v3_full.sh [N_SHARDS] [MODEL ...]   (default 8 shards, 9 gated jurors)
set -uo pipefail
T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis
C=$T/campaigns/allslides
F=$T/feasibility
N_SHARDS=${1:-8}; shift || true
MODELS=("$@"); [[ ${#MODELS[@]} -eq 0 ]] && MODELS=(gemma3:12b gemma3:27b phi4:14b qwen3:14b qwen3:32b deepseek-r1:14b mistral-small3.2 llama3.1:8b granite3.3:8b)
mkdir -p "$C/slurm"
big() { [[ $1 == *70b* || $1 == *120b* ]]; }
n_sub=0
for m in "${MODELS[@]}"; do
  tag=$(echo "$m" | tr ':.' '__')
  for s in $(seq 0 $((N_SHARDS-1))); do
    name="sec3full_${tag}_s${s}"
    [[ -f "$F/runs/$name/done.json" ]] && continue
    cg=gpu:1; big "$m" && cg=gpu:2
    for part in h200 cuda; do
      extra=""; gres=gpu:1
      [[ $part == h200 ]] && extra="--qos=h200_preempt --exclude=clust1-h200-1"
      [[ $part == cuda ]] && gres=$cg
      sbatch --parsable --job-name="$name" --partition="$part" $extra --gres="$gres" --cpus-per-task=8 --mem=64G --time=06:00:00 \
        --output="$C/slurm/${name}_${part}_%j.out" --error="$C/slurm/${name}_${part}_%j.err" \
        --export=ALL,TASK_NAME=$name,CONDA_ENV=pathology,MODEL=$m,SHARD=$s,N_SHARDS=$N_SHARDS,JSON_MODE=format,CONC=2,REQ_TIMEOUT=900,TASK_CMD="python $C/llm_grade_sections_v3.py" \
        --wrap="bash $F/run_task.sh" >/dev/null && n_sub=$((n_sub+1))
    done
  done
done
echo "submitted $n_sub jobs for ${#MODELS[@]} jurors x $N_SHARDS shards (x2 partitions)"
