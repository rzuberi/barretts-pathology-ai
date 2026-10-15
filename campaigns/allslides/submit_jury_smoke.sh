#!/usr/bin/env bash
# Jury v3 smoke: 11 cached candidates x 80 adjudicated reports, race-to-run on
# h200 + cuda (duplicate submissions; first to start owns the task, the other exits).
# Big models (70B/120B) need >48 GB: h200 gpu:1, cuda gpu:2 (ollama spans both L40S).
# gpt-oss additionally gets a JSON_MODE=none variant (it failed under format:json in v2).
set -uo pipefail
T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis
C=$T/campaigns/allslides
F=$T/feasibility
mkdir -p "$C/slurm"
SMALL=(gemma3:12b gemma3:27b phi4:14b qwen3:14b qwen3:32b deepseek-r1:14b mistral-small3.2 llama3.1:8b granite3.3:8b)
BIG=(llama3.3:70b gpt-oss:120b)
submit() {  # name model json_mode partition gres
  local name=$1 model=$2 jm=$3 part=$4 gres=$5 excl=""
  [[ $part == h200 ]] && excl="--qos=h200_preempt --exclude=clust1-h200-1"
  sbatch --parsable --job-name="$name" --partition="$part" $excl --gres="$gres" --cpus-per-task=8 --mem=64G --time=03:00:00 \
    --output="$C/slurm/${name}_${part}_%j.out" --error="$C/slurm/${name}_${part}_%j.err" \
    --export=ALL,TASK_NAME=$name,CONDA_ENV=pathology,MODEL=$model,SMOKE=adjudicated,JSON_MODE=$jm,CONC=2,REQ_TIMEOUT=900,TASK_CMD="python $C/llm_grade_sections_v3.py" \
    --wrap="bash $F/run_task.sh"
}
for m in "${SMALL[@]}"; do
  n="sec3smoke_$(echo "$m" | tr ':.' '__')"
  echo "$n: $(submit "$n" "$m" format h200 gpu:1) $(submit "$n" "$m" format cuda gpu:1)"
done
for m in "${BIG[@]}"; do
  n="sec3smoke_$(echo "$m" | tr ':.' '__')"
  echo "$n: $(submit "$n" "$m" format h200 gpu:1) $(submit "$n" "$m" format cuda gpu:2)"
done
n=sec3smoke_gpt-oss_120b_nojson
echo "$n: $(submit "$n" gpt-oss:120b none h200 gpu:1) $(submit "$n" gpt-oss:120b none cuda gpu:2)"
echo "analyse with: python $C/analyze_sections_smoke.py $F/runs"
