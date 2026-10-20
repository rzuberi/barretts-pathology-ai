#!/bin/bash
# rerun after the pack-reader fix: all 24 training units -> aggregate + figure 3. Log: chain.log
A=/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio; TH=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; L=$A/chain.log
log() { echo "$(date '+%a %H:%M') $*" >> $L; }
log "chain2 start (pack-reader fix)"
(cd $TH && bash scripts/cluster/campaign.sh ttrain2 $A/code/tasks_train.tsv --partitions "cuda h200" --workers 3 --gres gpu:1 --cpus 6 --mem 72G --time 1:30:00 --worker-time 4:00:00) >> $L 2>&1
bash $A/code/wait_campaign.sh ttrain2 >> $L 2>&1 || { log "STOP: ttrain2 incomplete"; exit 1; }
log "ttrain2 all done; preds $(ls $A/preds | grep -c npz)"
J=$(sbatch --parsable -p epyc -c 8 --mem 64G -t 1:00:00 -J tt_agg -o $A/slurm/agg_%j.out --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && python $A/code/aggregate_pub.py && python $A/code/fig3_overlay.py && python $A/code/gpu_hours.py && python $A/code/make_report.py $A/results"); log "aggregate job $J"
while squeue -h -j $J 2>/dev/null | grep -q .; do sleep 30; done
log "AGG DONE state=$(sacct -j $J -n -X -o State | tr -d ' ' | head -1)"
