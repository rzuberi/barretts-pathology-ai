#!/usr/bin/env bash
# Learning-curve GPU phase (follow-up 4): one race-to-run task per fold, arm B, SIZES env (default 150,500,1000), twinned h200 + cuda.
# Usage: submit_gpu.sh [afterok_jobid]
set -uo pipefail
A=/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun; T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; F=$T/feasibility
DEP=${1:-}; [[ -n "$DEP" ]] && DEP="--dependency=afterok:$DEP"
for f in 0 1 2 3 4; do
  name=isbi_lc$f; [[ -f $F/runs/$name/done.json ]] && continue
  for part in h200 cuda; do extra=""; [[ $part == h200 ]] && extra="--qos=h200_preempt"
    sbatch --parsable --job-name=$name --partition=$part $extra $DEP --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:30:00 \
      --output=$A/slurm/${name}_${part}_%j.out --error=$A/slurm/${name}_${part}_%j.err \
      --export=ALL,TASK_NAME=$name,CONDA_ENV=virchow2,FOLD=$f,SIZES=${SIZES:-150,500,1000},TASK_CMD="python $A/code/train_lc.py" --wrap="bash $F/run_task.sh"
  done
done | tr '\n' ' '; echo "<- fold jobs"
