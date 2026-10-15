#!/bin/bash
# usage: pool_submit.sh RUN_DIR [n1gpu=6] [n2gpu=1] [nh200=4]
RUN=$1; N1=${2:-6}; N2=${3:-1}; NH=${4:-4}
S=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/lit_triage/lit_pool.py
W="source /home/zuberi01/miniforge3/etc/profile.d/conda.sh; conda activate erin; RUN=$RUN NUM_CTX=${NUM_CTX:-8192} python $S"
mkdir -p $RUN/slurm
for i in $(seq 1 $N1); do sbatch -J litpool -p cuda --gres=gpu:1 -c 4 --mem=32G -t 3:00:00 -o $RUN/slurm/%j.out --wrap "$W"; done
for i in $(seq 1 $N2); do sbatch -J litpool -p cuda --gres=gpu:2 -c 4 --mem=64G -t 3:00:00 -o $RUN/slurm/%j.out --wrap "$W"; done
for i in $(seq 1 $NH); do sbatch -J litpool -p h200 --qos=h200_preempt --exclude=clust1-h200-1 --gres=gpu:1 -c 4 --mem=64G -t 3:00:00 -o $RUN/slurm/%j.out --wrap "$W"; done
