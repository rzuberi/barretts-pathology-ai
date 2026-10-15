#!/usr/bin/env bash
# Fan pull-workers across GPU partitions. Re-run any time: workers that land on an
# empty queue exit in seconds. Args: N_CUDA N_H200 [WALL hh:mm:ss]
set -uo pipefail
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/campaigns/allslides
MANIFEST=$C/erin_manifest_all_he.txt
OUTDIR=/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/features/20x_224px/features_uni_v2_all
N_CUDA=${1:-48}; N_H200=${2:-16}; WALL=${3:-03:00:00}
WALL_S=$(( $(echo "$WALL" | awk -F: '{print $1*3600+$2*60+$3}') ))
mkdir -p "$C/slurm" "$OUTDIR"
REMAINING=$(python3 - "$MANIFEST" "$OUTDIR" <<'EOF'
import sys, os
paths=[l.strip() for l in open(sys.argv[1]) if l.strip()]
print(sum(not os.path.exists(os.path.join(sys.argv[2], os.path.splitext(os.path.basename(p))[0]+".h5")) for p in paths))
EOF
)
echo "slides remaining: $REMAINING"
[[ "$REMAINING" -eq 0 ]] && { echo "nothing to do"; exit 0; }
COMMON=(--gres=gpu:1 --cpus-per-task=${CPUS:-4} --mem=${MEM:-32G} --time="$WALL" --output="$C/slurm/xw_%x_%j.out" --error="$C/slurm/xw_%x_%j.err")
# cuda / L40S: erin env (torch 2.0.1+cu117 has sm_89 kernels)
for i in $(seq 1 "$N_CUDA"); do
  sbatch --parsable --job-name=xw_cuda --partition=cuda "${COMMON[@]}" \
    --export=ALL,MANIFEST=$MANIFEST,OUTDIR=$OUTDIR,WALL_SECONDS=$WALL_S \
    --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate erin && python $C/extract_worker.py"
done | tr '\n' ' '; echo " <- cuda workers"
# h200: virchow2 env (torch 2.9.1+cu128, sm_90); clust1-h200-1 excluded (ECC faults); preemptible -> idempotent units
for i in $(seq 1 "$N_H200"); do
  sbatch --parsable --job-name=xw_h200 --partition=h200 --qos=h200_preempt --exclude=clust1-h200-1 "${COMMON[@]}" \
    --export=ALL,MANIFEST=$MANIFEST,OUTDIR=$OUTDIR,WALL_SECONDS=$WALL_S \
    --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate virchow2 && python $C/extract_worker.py"
done | tr '\n' ' '; echo " <- h200 workers"
