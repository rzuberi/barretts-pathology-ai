#!/bin/bash
# Cluster-side chain for the Panakeia probe (login node: polling + sbatch only). Log: outreach_panakeia/chain.log
# training smoke (1 ABMIL + 1 mean-pool unit) -> stop if it fails -> all 63 training units -> aggregate -> heatmaps.
TH=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; A=/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia; L=$A/chain.log; RUNS=$TH/feasibility/runs; C=$A/code
log() { echo "$(date +%H:%M) $*" >> $L; }
cd $TH; log "chain start (v2: epyc twins for training)"; mkdir -p $A/slurm
grep -P "^(msi_strat_abmil_r0|wgd_strat_meanlr_r0)\t" $C/op_train.tsv > $C/op_trsmoke.tsv
bash scripts/cluster/campaign.sh op_trsmoke $C/op_trsmoke.tsv --partitions "cuda h200" --per-task yes --gres gpu:1 --cpus 4 --workers 0 --mem 32G --time 1:00:00 >> $L 2>&1
bash scripts/cluster/campaign.sh op_trsmoke $C/op_trsmoke.tsv --partitions "epyc" --per-task yes --cpus 8 --workers 0 --mem 32G --time 1:30:00 >> $L 2>&1
for t in msi_strat_abmil_r0 wgd_strat_meanlr_r0; do
  while [ ! -f $RUNS/op_trsmoke__$t/done.json ]; do sleep 60
    if ! squeue -h -u $USER -n op_trsmoke__$t | grep -q .; then sleep 30; [ -f $RUNS/op_trsmoke__$t/done.json ] || { log "SMOKE FAILED ($t) - chain stops"; exit 1; }; fi; done
  n=$(($(wc -l < $A/preds/$t.csv) - 1)); log "smoke $t done: $n predictions; $(tr -d '\n ' < $A/preds/$t.log.json | head -c 400)"
  [ $n -ge 300 ] || { log "SMOKE: too few predictions - chain stops"; exit 1; }
done
bash scripts/cluster/campaign.sh op_train $C/op_train.tsv --partitions "cuda h200" --per-task yes --gres gpu:1 --cpus 4 --workers 2 --mem 32G --time 1:00:00 --worker-time 6:00:00 >> $L 2>&1
bash scripts/cluster/campaign.sh op_train $C/op_train.tsv --partitions "epyc" --per-task yes --cpus 8 --workers 4 --mem 32G --time 1:30:00 --worker-time 8:00:00 >> $L 2>&1
while :; do d=$(ls $A/preds/*.csv 2>/dev/null | wc -l); log "TRAIN $d/63"; [ $d -ge 63 ] && break; sleep 300; done
for s in aggregate heatmaps; do
  J=$(sbatch --parsable --partition=epyc --time=01:00:00 --cpus-per-task=4 --mem=32G --job-name=op_$s --output=$A/slurm/${s}_%j.out --error=$A/slurm/${s}_%j.err --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate virchow2 && python $C/$s.py"); log "$s job $J"
  while squeue -h -j $J -o %T | grep -q .; do sleep 30; done; log "$s DONE state=$(sacct -j $J -n -X -o State | tr -d ' ' | head -1)"
done
log "chain complete"
