#!/usr/bin/env bash
# Post-maintenance reconnect runbook (written 2026-08-18, for Wed 19th evening).
# Run FROM THE MAC: bash scripts/reconnect.sh          -> report state only
#                   bash scripts/reconnect.sh --resume -> also release/resubmit
set -uo pipefail
MODE="${1:-report}"

T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis
C=$T/campaigns
ERIN_F=/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/features/20x_224px/features_uni_v2
TCGA_F=/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2
TCGA_S=/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/slides

ssh -o BatchMode=yes -o ConnectTimeout=15 cluster bash -s -- "$MODE" <<'REMOTE'
MODE="$1"
T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis
C=$T/campaigns
echo "=== connected: $(hostname), $(date) ==="

echo "=== 1. repo sync ==="
cd $T && git pull --ff-only 2>&1 | tail -2

echo "=== 2. queue state ==="
Q=$(squeue -u zuberi01 -h 2>/dev/null | wc -l)
echo "jobs in queue: $Q"
squeue -u zuberi01 -h -o "%.14i %.6P %.16j %.2t %r" 2>/dev/null | head -8

echo "=== 3. campaign progress ==="
E=$(ls /mnt/scratche/fast/fmlab/datasets/imaging/ERIN/features/20x_224px/features_uni_v2/*.h5 2>/dev/null | wc -l)
TX=$(ls /mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2/*.h5 2>/dev/null | wc -l)
TS=$(ls /mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/slides/*.svs 2>/dev/null | wc -l)
echo "ERIN features: $E/2281 | TCGA slides: $TS/65 | TCGA features: $TX/65"

echo "=== 4. OCCAMS v2 (read against docs/occams_v2_decision_tree.md) ==="
cat $C/runs/occams_fusion_v2/output/results.json 2>/dev/null || echo "NO RESULTS — job never ran or was purged"

if [ "$MODE" = "--resume" ]; then
  echo "=== 5. resume ==="
  # held arrays survive a reboot as held; release is a no-op if they're gone
  scontrol release 56401367 56401368 2>&1 | head -1
  if [ "$Q" -eq 0 ]; then
    echo "queue empty -> resubmitting (idempotent: done-markers/h5 skip finished units)"
    [ "$E" -lt 2281 ] && bash $C/submit_erin_extraction.sh 2>&1 | tail -1
    [ "$TS" -lt 65 ] && bash $C/submit_tcga_acquisition.sh 2>&1 | tail -1
    [ ! -f $C/runs/occams_fusion_v2/done.json ] && bash $C/submit_occams_v2.sh 2>&1 | tail -1
    echo "resubmission attempted — if a submit script name above doesn't exist, ls $C and use the actual name"
  else
    echo "queue non-empty -> released holds only; let existing jobs drain first"
  fi
fi
echo "=== done ==="
REMOTE
