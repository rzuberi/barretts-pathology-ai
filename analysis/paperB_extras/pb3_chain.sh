#!/bin/bash
# Cluster-side chain (login node, polling only): read campaign -> GPU score campaign -> evaluation job. Logs to chain.log.
TH=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; O=/mnt/scratche/slow/fmlab/zuberi01/analysis/paperB_extras; L=$O/chain.log
log() { echo "$(date +%H:%M) $*" >> $L; }
log "chain start"
while :; do d=$(ls $TH/feasibility/runs/pb3read__rd*/done.json 2>/dev/null | wc -l); log "READ $d/392 tiles=$(ls $O/zeroshot/tiles | wc -l)"; [ $d -ge 392 ] && break; sleep 180; done
log "read phase complete; tiles dir $(du -sh $O/zeroshot/tiles | cut -f1)"
cd $TH && bash scripts/cluster/campaign.sh pb3score $O/code/pb3score.tsv --partitions "cuda h200" --workers 2 --per-task yes --gres gpu:1 --cpus 8 --mem 32G --time 1:00:00 --worker-time 6:00:00 >> $L 2>&1
while :; do d=$(ls $TH/feasibility/runs/pb3score__sc*/done.json 2>/dev/null | wc -l); log "SCORE $d/98 conch=$(ls $O/zeroshot/conch 2>/dev/null | wc -l)"; [ $d -ge 98 ] && break; sleep 120; done
log "score phase complete"
J=$(sbatch --parsable --partition=epyc --time=01:00:00 --cpus-per-task=4 --mem=32G --job-name=paperB_3 --output=$O/slurm/run3_%j.out --error=$O/slurm/run3_%j.err --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && MODE=subset python $O/code/paperB_3.py && MODE=attn64 python $O/code/paperB_3.py")
log "eval job $J submitted"
while squeue -h -j $J -o %T | grep -q .; do sleep 60; done
log "EVAL DONE state=$(sacct -j $J -n -X -o State | tr -d ' ' | head -1)"
