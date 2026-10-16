#!/bin/bash
# Project 2 chain: wait for both P1 RL selections -> pick the better on validation -> merge -> conf baseline + 3 alpha runs -> ckpt evals -> best -> metrics.
T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; C=$T/scripts/rl; RUNS=$T/feasibility/runs; R=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_grader; A=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_abstain
L=$A/chain.log; P2=$C/full_p2.sh; source ~/miniforge3/etc/profile.d/conda.sh; conda activate pathology
log() { echo "$(date +%H:%M) $*" >> $L; }; done_() { [ -f $RUNS/$1/done.json ]; }; waitfor() { while ! done_ $1; do sleep 180; done; }
race() { done_ $1 || bash $C/race3.sh "$@" >/dev/null; }; ckpts() { ls -d $1/checkpoint-* 2>/dev/null | xargs -n1 basename | sort -t- -k2 -n; }
log "p2 chain start"; waitfor p1_best_grpo_base; waitfor p1_best_grpo_sft
pick=$(python - <<PY
import json
a=json.load(open("$R/selected_grpo_base.json")); b=json.load(open("$R/selected_grpo_sft.json"))
sa=a["val_scores"][a["selected"]]; sb=b["val_scores"][b["selected"]]
print(("grpo_base", a["selected"]) if sa>=sb else ("grpo_sft", b["selected"]))
PY
)
run=$(echo $pick | tr -d "(),'" | cut -d" " -f1); ck=$(echo $pick | tr -d "(),'" | cut -d" " -f2); log "P1 RL model for project 2: $run $ck"
base=$(ls -d /mnt/scratche/slow/fmlab/zuberi01/hf_cache/hub/models--google--medgemma-4b-it/snapshots/*/ | head -1); [ $run = grpo_sft ] && base=$R/runs/sft_merged
echo "{\"run\": \"$run\", \"checkpoint\": \"$ck\"}" > $R/p1_best.json
srun -p epyc -t 1:00:00 -c 8 --mem=96G -J p1merge bash -c "source ~/miniforge3/etc/profile.d/conda.sh; conda activate /mnt/scratche/slow/fmlab/zuberi01/envs/rl2; python $C/merge.py $base $R/runs/$run/$ck $R/runs/p1_best_merged" >> $L 2>&1
log "merged"; race p2_conf conf 4:00:00 $P2; for a in 0.2 0.4 0.6; do race p2_train_$a "train $a" 14:00:00 $P2; done
for a in 0.2 0.4 0.6; do ( waitfor p2_train_$a; log "alpha $a trained"
  for c in $(ckpts $A/runs/alpha$a); do race p2_ck_${a}_$c "ckpt $a $c" 2:00:00 $P2; done; for c in $(ckpts $A/runs/alpha$a); do waitfor p2_ck_${a}_$c; done
  b=$(python $C/select_best.py p2 alpha$a); log "alpha $a best $b"; race p2_best_$a "best $a $b" 4:00:00 $P2; waitfor p2_best_$a; log "alpha $a best evaluated" ) & done
wait; waitfor p2_conf
tags=$(for a in 0.2 0.4 0.6; do python -c "import json; print('alpha$a__'+json.load(open('$A/selected_alpha$a.json'))['selected'])"; done | tr "\n" "," | sed "s/,$//")
best=$(python - <<PY
import json
s={a: json.load(open(f"$A/selected_alpha{a}.json")) for a in ("0.2","0.4","0.6")}
k=max(s, key=lambda a: s[a]["val_scores"][s[a]["selected"]]); print(f"alpha{k}__"+s[k]["selected"])
PY
)
log "best alpha on validation: $best"; cd $C; ALPHA_TAGS=$tags BEST=$best OUT=$A/metrics.json srun -p epyc -t 1:00:00 -c 2 --mem=24G -J p2met python p2_metrics.py >> $L 2>&1; log "P2 CHAIN DONE"
