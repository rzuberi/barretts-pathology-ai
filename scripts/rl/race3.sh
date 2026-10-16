#!/bin/bash
# race3.sh NAME "STAGE ARGS" [TIME] [SCRIPT]: one task raced on cuda, h200 and the clust1-h200-1 node; first to start cancels the rest.
T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; E=/mnt/scratche/slow/fmlab/zuberi01/envs/rl2; NAME=$1; ST=$2; TM=${3:-12:00:00}; SCR=${4:-$T/scripts/rl/full_p1.sh}
L=$T/feasibility/runs/_campaigns/rl; mkdir -p $L; cd $T
[ -f feasibility/runs/$NAME/done.json ] && { echo "$NAME already done"; exit 0; }
W="cd $T && TASK_NAME=$NAME CONDA_ENV=$E TASK_CMD=\"bash $SCR $ST\" bash feasibility/run_task.sh"
sbatch --parsable -p cuda --gres=gpu:1 -c 8 --mem=96G -t $TM -J $NAME -o $L/%x_%j.log --wrap="$W"
sbatch --parsable -p h200 --qos=h200_preempt --exclude=clust1-h200-1 --gres=gpu:1 -c 8 --mem=96G -t $TM -J $NAME -o $L/%x_%j.log --wrap="$W"
sbatch --parsable -p h200 --qos=h200_preempt --nodelist=clust1-h200-1 --gres=gpu:1 -c 8 --mem=96G -t $TM -J $NAME -o $L/%x_%j.log --wrap="$W"
