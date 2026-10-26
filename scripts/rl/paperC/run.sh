#!/bin/bash
# Paper C CPU stages. Usage: bash run.sh STAGE  (shift | corevote | score). Submit with sbatch on epyc; logs in $O/logs.
set -eo pipefail
B=/mnt/scratche/slow/fmlab/zuberi01; C=$B/phd/thesis/scripts/rl/paperC; O=$B/analysis/paperC; mkdir -p $O/logs $O/preds
export PYTHONPATH=$C OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PY=$B/envs/rl2/bin/python
case $1 in
  shift)    $PY $C/shift_preds.py ;;
  corevote) $PY $C/core_voting.py ;;
  score)    $PY $C/score_all.py ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
