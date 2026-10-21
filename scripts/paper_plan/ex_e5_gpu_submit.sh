#!/usr/bin/env bash
# E5 GPU submitter: every task of TASKS.tsv (name<TAB>conda_env<TAB>command) raced as an h200 job (--qos=h200_preempt --exclude=clust1-h200-1) and a cuda
# twin, both --gres=gpu:1, through feasibility/run_task.sh (done marker + lock; the first to start cancels its pending twin).
# usage: bash ex_e5_gpu_submit.sh PREFIX TASKS.tsv [time] [mem] [cpus] [dependency]
set -uo pipefail
TH=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; P=$1; TASKS=$2; TIME=${3:-2:00:00}; MEM=${4:-64G}; CPUS=${5:-8}; DEP=${6:-}
D=$TH/feasibility/runs/_campaigns/$P; mkdir -p $D; cp $TASKS $D/tasks.tsv; [ -n "$DEP" ] && DEP="--dependency=afterok:$DEP"
while IFS=$'\t' read -r name env cmd; do
  [ -z "$name" ] && continue
  for part in h200 cuda; do extra=""; [ $part = h200 ] && extra="--qos=h200_preempt --exclude=clust1-h200-1"
    sbatch --parsable -p $part $extra $DEP --gres=gpu:1 -c $CPUS --mem=$MEM -t $TIME -J ${P}__$name -o $D/task_%x_%j.log \
      --wrap="cd $TH && TASK_NAME=${P}__$name CONDA_ENV='$env' TASK_CMD='$cmd' bash feasibility/run_task.sh"
  done
done < $D/tasks.tsv | tr '\n' ' '; echo "<- $P jobs"
