#!/bin/bash
# Cluster-side chain for the tile-budget study (login node; polling + sbatch only). Log: tilebudget/chain.log
TH=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; A=/mnt/scratche/slow/fmlab/zuberi01/analysis/tilebudget; L=$A/chain.log; RUNS=$TH/feasibility/runs
log() { echo "$(date +%H:%M) $*" >> $L; }
PARTS="cuda h200"
cd $TH; log "chain start"
# P0: wait for the pack, finalize
while :; do d=$(ls $RUNS/tbpack__pk*/done.json 2>/dev/null | wc -l); log "PACK $d/392"; [ $d -ge 392 ] && break; sleep 120; done
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && python $A/code/finalize_pack_all.py >> $L 2>&1
# P1 fold 0 then P2 smoke (fold 0, rk16)
bash scripts/cluster/campaign.sh tbsel0 $A/code/tb_sel_f0.tsv --partitions "$PARTS" --per-task yes --gres gpu:1 --cpus 8 --workers 0 --mem 140G --time 2:00:00 >> $L 2>&1
while [ ! -f $RUNS/tbsel0__sel_f0/done.json ]; do sleep 60; done; log "P1 fold0 done"
bash scripts/cluster/campaign.sh tbsmoke $A/code/tb_smoke.tsv --partitions "$PARTS" --per-task yes --gres gpu:1 --cpus 8 --workers 0 --mem 64G --time 1:30:00 >> $L 2>&1
while [ ! -f $RUNS/tbsmoke__f0_rk16/done.json ]; do sleep 60; if ! squeue -h -u $USER -n tbsmoke__f0_rk16 | grep -q .; then [ -f $RUNS/tbsmoke__f0_rk16/done.json ] || { log "SMOKE FAILED - chain stops"; exit 1; }; fi; done; log "P2 smoke done"
# remaining P1 folds, then all P2 units (workers funnel; units whose sel file is missing fail fast and are retried by the workers' second pass)
bash scripts/cluster/campaign.sh tbsel $A/code/tb_sel.tsv --partitions "$PARTS" --per-task yes --gres gpu:1 --cpus 8 --workers 0 --mem 140G --time 2:00:00 >> $L 2>&1
while :; do d=$(ls $RUNS/tbsel__sel_f*/done.json 2>/dev/null | wc -l); log "SEL $d/5"; [ $d -ge 5 ] && break; sleep 120; done
bash scripts/cluster/campaign.sh tbtrain $A/code/tb_train.tsv --partitions "$PARTS" --per-task yes --gres gpu:1 --cpus 8 --workers 3 --mem 64G --time 1:30:00 --worker-time 8:00:00 >> $L 2>&1
bash scripts/cluster/campaign.sh tbtrainall $A/code/tb_trainall.tsv --partitions "$PARTS" --per-task yes --gres gpu:1 --cpus 8 --workers 2 --mem 140G --time 3:00:00 --worker-time 10:00:00 >> $L 2>&1
while :; do d=$(ls $RUNS/tbtrain__f*/done.json $RUNS/tbtrainall__f*/done.json 2>/dev/null | wc -l); log "TRAIN $d/45"; [ $d -ge 45 ] && break; sleep 300; done
log "all units done"
J=$(sbatch --parsable --partition=epyc --time=02:00:00 --cpus-per-task=8 --mem=64G --job-name=tb_agg --output=$A/slurm/agg_%j.out --error=$A/slurm/agg_%j.err --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && python $A/code/aggregate_tb.py"); log "AGG job $J"
while squeue -h -j $J -o %T | grep -q .; do sleep 60; done; log "AGG DONE state=$(sacct -j $J -n -X -o State | tr -d ' ' | head -1)"
