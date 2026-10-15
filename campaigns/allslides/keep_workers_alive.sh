#!/usr/bin/env bash
# Self-rescheduling sweeper: keeps the extraction worker queue topped up until the
# manifest is exhausted. Runs on epyc (1 CPU, seconds of work), then resubmits
# itself with --begin=now+INTERVAL. Stops (no resubmit) when nothing remains.
# Start once:  sbatch --partition=epyc --time=00:10:00 --cpus-per-task=1 --mem=2G \
#              --job-name=xw_sweeper --output=.../slurm/sweeper_%j.out keep_workers_alive.sh
set -uo pipefail
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/campaigns/allslides
MANIFEST=$C/erin_manifest_all_he.txt
OUTDIR=/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/features/20x_224px/features_uni_v2_all
TARGET_CUDA=${TARGET_CUDA:-16}; TARGET_H200=${TARGET_H200:-8}; INTERVAL=${INTERVAL:-30}   # minutes
[[ -f $C/SWEEPER_STOP ]] && { echo "stop file present; not rescheduling"; exit 0; }
REMAINING=$(python3 - "$MANIFEST" "$OUTDIR" <<'EOF'
import sys, os
paths=[l.strip() for l in open(sys.argv[1]) if l.strip()]
print(sum(not os.path.exists(os.path.join(sys.argv[2], os.path.splitext(os.path.basename(p))[0]+".h5")) for p in paths))
EOF
)
PD_CUDA=$(squeue -h -u zuberi01 -t PD -n xw_cuda | wc -l); PD_H200=$(squeue -h -u zuberi01 -t PD -n xw_h200 | wc -l)
R_CUDA=$(squeue -h -u zuberi01 -t R -n xw_cuda | wc -l);   R_H200=$(squeue -h -u zuberi01 -t R -n xw_h200 | wc -l)
echo "$(date -Iseconds) remaining=$REMAINING cuda R/PD=$R_CUDA/$PD_CUDA h200 R/PD=$R_H200/$PD_H200"
if [[ $REMAINING -eq 0 ]]; then echo "manifest exhausted — sweeper retiring"; date -Iseconds > $C/EXTRACTION_COMPLETE; exit 0; fi
# Failure floor: retire if the remaining count has not moved for STALL_LIMIT cycles.
# Without this the pool is refilled forever against permanently-failing slides
# (2026-09-22: 18 unreadable/oversized slides burned 1,093 OOM-killed workers in 24 h).
STALL_LIMIT=${STALL_LIMIT:-3}
PREV=$(cat $C/.sweeper_prev_remaining 2>/dev/null || echo -1)
STALL=$(cat $C/.sweeper_stall 2>/dev/null || echo 0)
if [[ "$REMAINING" -eq "$PREV" ]]; then STALL=$((STALL+1)); else STALL=0; fi
echo "$REMAINING" > $C/.sweeper_prev_remaining; echo "$STALL" > $C/.sweeper_stall
if [[ $STALL -ge $STALL_LIMIT ]]; then
  echo "STALLED at $REMAINING slides for $STALL cycles — sweeper retiring; inspect slurm/*.err"
  date -Iseconds > $C/SWEEPER_STALLED; exit 0
fi
need_c=$(( TARGET_CUDA - PD_CUDA )); need_h=$(( TARGET_H200 - PD_H200 ))
(( need_c > 0 )) && bash $C/submit_extract_workers.sh $need_c 0 02:00:00 | tail -1
(( need_h > 0 )) && bash $C/submit_extract_workers.sh 0 $need_h 01:00:00 | tail -1
sbatch --parsable --partition=epyc --time=00:10:00 --cpus-per-task=1 --mem=2G --job-name=xw_sweeper \
  --begin=now+${INTERVAL}minutes --output=$C/slurm/sweeper_%j.out --error=$C/slurm/sweeper_%j.err \
  --export=ALL,TARGET_CUDA=$TARGET_CUDA,TARGET_H200=$TARGET_H200,INTERVAL=$INTERVAL "$C/keep_workers_alive.sh" \
  | sed 's/^/rescheduled as job /'
