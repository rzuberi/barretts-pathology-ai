#!/bin/bash
# wait for the 15 extra learning-curve units, then rerun the zero-shot/label-scale evaluation (both tile modes)
TH=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; O=/mnt/scratche/slow/fmlab/zuberi01/analysis/paperB_extras; L=$O/chain2.log
log() { echo "$(date +%H:%M) $*" >> $L; }
while :; do d=$(ls $TH/feasibility/runs/pb3lc2__lc_f*/done.json 2>/dev/null | wc -l); log "LC2 $d/15"; [ $d -ge 15 ] && break; sleep 120; done
J=$(sbatch --parsable --partition=epyc --time=02:00:00 --cpus-per-task=4 --mem=32G --job-name=paperB_3v2 --output=$O/slurm/run3v2_%j.out --error=$O/slurm/run3v2_%j.err --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && MODE=subset python $O/code/paperB_3.py && MODE=attn64 python $O/code/paperB_3.py")
log "EVAL3 job $J"; while squeue -h -j $J -o %T | grep -q .; do sleep 60; done; log "EVAL3 DONE state=$(sacct -j $J -n -X -o State | tr -d ' ' | head -1)"
