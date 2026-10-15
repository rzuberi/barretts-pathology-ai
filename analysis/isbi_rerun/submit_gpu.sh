#!/usr/bin/env bash
# ISBI rerun GPU phase: one race-to-run task per fold (all arms x tasks x seeds x holdouts inside), twinned h200 + cuda.
# Usage: submit_gpu.sh [afterok_jobid]
set -uo pipefail
A=/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun; T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; F=$T/feasibility
DEP=${1:-}; [[ -n "$DEP" ]] && DEP="--dependency=afterok:$DEP"
for f in 0 1 2 3 4; do
  name=isbi_fold$f; [[ -f $F/runs/$name/done.json ]] && continue
  for part in h200 cuda; do extra=""; [[ $part == h200 ]] && extra="--qos=h200_preempt"
    sbatch --parsable --job-name=$name --partition=$part $extra $DEP --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 \
      --output=$A/slurm/${name}_${part}_%j.out --error=$A/slurm/${name}_${part}_%j.err \
      --export=ALL,TASK_NAME=$name,CONDA_ENV=virchow2,FOLD=$f,TASK_CMD="python $A/code/train_fold.py" --wrap="bash $F/run_task.sh"
  done
done | tr '\n' ' '; echo "<- fold jobs"
