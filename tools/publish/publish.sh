#!/bin/bash
# Sync the public code mirror: stage + scan on the cluster (blocks on any identifier, report text, secret or large file),
# copy the staged code here, commit as rzuberi and push. Usage: tools/publish/publish.sh "commit message"
set -euo pipefail
cd "$(dirname "$0")/../.."; MSG=${1:-"sync code from cluster"}
C=/mnt/scratche/slow/fmlab/zuberi01
scp -q tools/publish/stage_and_scan.py cluster:$C/public_repo_stage_and_scan.py
ssh -o BatchMode=yes cluster "source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && srun --partition=epyc --time=00:30:00 --cpus-per-task=2 --mem=24G --job-name=pubscan python $C/public_repo_stage_and_scan.py" \
  || { echo "SCAN FAILED: nothing published. See $C/public_repo_scan.json on the cluster."; exit 1; }
# code folders are mirrored exactly (deletions propagate); hand-written docs at the repo root and tools/ are untouched
for d in labeller campaigns scripts feasibility analysis; do rsync -a --delete "cluster:$C/public_repo_stage/$d/" "$d/"; done
# local guard: refuse anything big or data-like even if it slipped through
big=$(find labeller campaigns scripts feasibility analysis -type f \( -size +1M -o -name "*.csv" -o -name "*.json" -o -name "*.npz" -o -name "*.h5" -o -name "*.pt" -o -name "*.db" -o -name "*.sqlite" -o -name "*.xlsx" -o -name "*.parquet" -o -name "*.tsv" \) | head -5)
[ -z "$big" ] || { echo "LOCAL GUARD: data-like or large files present: $big"; exit 1; }
git add -A
git diff --cached --quiet && { echo "no changes"; exit 0; }
git -c user.name=rzuberi -c user.email=56508673+rzuberi@users.noreply.github.com commit -q -m "$MSG"
git push -q origin HEAD && git log -1 --format="pushed %h %s"
