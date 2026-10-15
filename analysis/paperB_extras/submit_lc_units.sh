#!/usr/bin/env bash
# Learning curve: 15 small race-to-run units (fold x training-patient size), each twinned on h200 + cuda; a unit trains
# 2 tasks x 3 seeds = 6 arm-B models with the dataset loaded once. A twin that finds done.json (or the lock) exits at once.
set -uo pipefail
A=/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun; T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; F=$T/feasibility
for f in 0 1 2 3 4; do for n in 150 500 1000; do
  name=isbi_lc_f${f}_n${n}; [[ -f $F/runs/$name/done.json ]] && continue
  for part in h200 cuda; do extra=""; [[ $part == h200 ]] && extra="--qos=h200_preempt"
    sbatch --parsable --job-name=$name --partition=$part $extra --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:40:00 \
      --output=$A/slurm/${name}_${part}_%j.out --error=$A/slurm/${name}_${part}_%j.err \
      --export=ALL,TASK_NAME=$name,CONDA_ENV=virchow2,FOLD=$f,SIZES=$n,TASK_CMD="python $A/code/train_lc.py" --wrap="bash $F/run_task.sh"
  done
done; done
