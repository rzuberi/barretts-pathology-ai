#!/usr/bin/env bash
set -uo pipefail
A=/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun; C=$A/code; T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; F=$T/feasibility
for j in isbi_fold0 isbi_fold1 isbi_fold2 isbi_fold3 isbi_fold4 isbi_part3; do scancel -u zuberi01 -n $j; done; sleep 3
fids=()
for f in 0 1 2 3 4; do name=isbi_fold$f; rm -rf $F/runs/$name
  for part in h200 cuda; do extra=""; [[ $part == h200 ]] && extra="--qos=h200_preempt"
    fids+=($(sbatch --parsable --job-name=$name --partition=$part $extra --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 --output=$A/slurm/${name}_${part}_%j.out --error=$A/slurm/${name}_${part}_%j.err --export=ALL,TASK_NAME=$name,CONDA_ENV=virchow2,FOLD=$f,TASK_CMD="python $C/train_fold.py" --wrap="bash $F/run_task.sh")); done; done
fdep=$(IFS=:; echo "${fids[*]}"); rm -rf $F/runs/isbi_part3
for part in h200 cuda; do extra=""; [[ $part == h200 ]] && extra="--qos=h200_preempt"
  sbatch --parsable --job-name=isbi_part3 --partition=$part $extra --dependency=afterok:$fdep --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 --output=$A/slurm/part3_${part}_%j.out --error=$A/slurm/part3_${part}_%j.err --export=ALL,TASK_NAME=isbi_part3,CONDA_ENV=virchow2,TASK_CMD="python $C/part3_swg.py" --wrap="bash $F/run_task.sh" >/dev/null; done
echo "resubmitted ${#fids[@]} fold jobs + part3 x2"; squeue -h -u zuberi01 -o "%j %P %t" | grep isbi | sort | uniq -c
