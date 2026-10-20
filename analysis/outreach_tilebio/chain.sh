#!/bin/bash
# Cluster-side chain (login node: polling + sbatch only). Log: outreach_tilebio/chain.log
A=/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio; TH=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; L=$A/chain.log; W=$A/code/wait_campaign.sh
log() { echo "$(date '+%a %H:%M') $*" >> $L; }
camp() { (cd $TH && bash scripts/cluster/campaign.sh "$@") >> $L 2>&1; }
need() { bash $W $1 >> $L 2>&1 || { log "STOP: campaign $1 ended without all tasks done"; exit 1; }; log "$1 all done"; }
log "chain start"
# ---- TCGA: pack -> train ----
need tpack_tcga
grep "^tcga_" $A/code/tasks_train.tsv > $A/code/tasks_train_tcga.tsv
camp ttrain_tcga $A/code/tasks_train_tcga.tsv --partitions "cuda h200" --workers 2 --gres gpu:1 --cpus 6 --mem 72G --time 1:30:00 --worker-time 4:00:00
# ---- CAMELYON16: rescue tiles -> embed the missing slides -> pack -> train ----
need ttrescue
awk '$1=="c16"' $A/code/embed_list.txt > $A/code/embed_list_c16.txt
printf "c16a\tvirchow2\tpython $A/code/embed_slide.py --pull $A/code/embed_list_c16.txt --wall 3400\nc16b\tvirchow2\tpython $A/code/embed_slide.py --pull $A/code/embed_list_c16.txt --wall 3400\n" > $A/code/tasks_embed_c16rescue.tsv
camp temb_c16r $A/code/tasks_embed_c16rescue.tsv --partitions "cuda h200" --workers 0 --gres gpu:1 --cpus 6 --mem 48G --time 1:00:00
need temb_c16r
n=$(ls $A/data/feats/uni2h/c16 | grep -c npy$); m=$(ls $A/data/feats/virchow2/c16 | grep -c npy$); log "c16 features uni=$n v2=$m"
[ $n -eq 399 ] && [ $m -eq 399 ] || { log "STOP: c16 features incomplete"; exit 1; }
grep "^pack_c16" $A/code/tasks_pack.tsv > $A/code/tasks_pack_c16.tsv
camp tpack_c16 $A/code/tasks_pack_c16.tsv --partitions "epyc rocm" --workers 1 --cpus 4 --mem 32G --time 1:00:00
need tpack_c16
grep "^c16_" $A/code/tasks_train.tsv > $A/code/tasks_train_c16.tsv
camp ttrain_c16 $A/code/tasks_train_c16.tsv --partitions "cuda h200" --workers 1 --gres gpu:1 --cpus 6 --mem 48G --time 1:30:00 --worker-time 3:00:00
J=$(sbatch --parsable -p epyc -c 2 --mem 8G -t 0:30:00 -J tt_lesion -o $A/slurm/lesion_%j.out --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && python $A/code/lesion_geom.py"); log "lesion job $J"
need ttrain_c16
need ttrain_tcga
while squeue -h -j $J 2>/dev/null | grep -q .; do sleep 30; done
J=$(sbatch --parsable -p epyc -c 8 --mem 64G -t 1:00:00 -J tt_agg -o $A/slurm/agg_%j.out --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && python $A/code/aggregate_pub.py && python $A/code/fig3_overlay.py"); log "aggregate job $J"
while squeue -h -j $J 2>/dev/null | grep -q .; do sleep 30; done
log "AGG DONE state=$(sacct -j $J -n -X -o State | tr -d ' ' | head -1)"
