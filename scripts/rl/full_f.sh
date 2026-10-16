#!/bin/bash
# Task F stages (PerturbQA). Never edit while jobs run.
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 30000))
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl; F=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_followups/F; R=$F/runs; mkdir -p $R
BASE=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
VS() { [ $1 = nor ] && echo val_nor || echo val; }
case $1 in
  f_zs)     BASE=$BASE TAG=zs SETS=val,val_nor,test python $C/pert_eval.py ;;
  f_sft)    MODE=sft BASE=$BASE OUT=$R/sft_$2 DATA=$F/data/train_$2.jsonl STEPS=3000 python $C/pert_train.py ;;
  f_sftck)  BASE=$BASE ADAPTER=$R/sft_$2/$3 TAG=sft_$2__$3 SETS=$(VS $2) python $C/pert_eval.py ;;
  f_merge)  python $C/merge.py $BASE $R/sft_$2/$3 $R/sft_$2_merged ;;
  f_grpo)   MODE=grpo BASE=$R/sft_$2_merged OUT=$R/grpo_$2 DATA=$F/data/grpo_$2.jsonl STEPS=300 python $C/pert_train.py ;;
  f_grpock) BASE=$R/sft_$2_merged ADAPTER=$R/grpo_$2/$3 TAG=grpo_$2__$3 SETS=$(VS $2) python $C/pert_eval.py ;;
  f_testsft)  X=""; [ $2 = nor ] && X="CELLS=rpe1"; env $X BASE=$BASE ADAPTER=$R/sft_$2/$3 TAG=sft_$2__$3 SETS=test python $C/pert_eval.py ;;
  f_testgrpo) X=""; [ $2 = nor ] && X="CELLS=rpe1"; env $X BASE=$R/sft_$2_merged ADAPTER=$R/grpo_$2/$3 TAG=grpo_$2__$3 SETS=test python $C/pert_eval.py ;;
  f_smoke)  SM=$F/smoke; rm -rf $SM; mkdir -p $SM
            BASE=$BASE TAG=smk_zs SETS=val N=200 python $C/pert_eval.py 2>&1 | grep -E "items in|Error" | tail -2
            MODE=sft BASE=$BASE OUT=$SM/sft DATA=$F/data/train_all.jsonl STEPS=10 N=200 python $C/pert_train.py 2>&1 | grep -E "TRAIN DONE|Error" | tail -2
            python $C/merge.py $BASE $SM/sft/final $SM/merged 2>&1 | tail -1
            MODE=grpo BASE=$SM/merged OUT=$SM/grpo DATA=$F/data/grpo_all.jsonl STEPS=3 N=64 python $C/pert_train.py 2>&1 | grep -E "'reward'|TRAIN DONE|Error" | tail -3
            BASE=$SM/merged ADAPTER=$SM/grpo/final TAG=smk_grpo SETS=val N=200 python $C/pert_eval.py 2>&1 | grep -E "items in|Error" | tail -2
            python - <<PY
import json
for t in ("smk_zs", "smk_grpo"):
    r = [json.loads(l) for l in open("$F/preds/" + t + "__val.jsonl")]
    acc = sum(max(("up","down","none"), key=lambda w: x["p_" + w]) == x["label"] for x in r) / len(r); mass = sum(x["mass"] for x in r) / len(r)
    print(t, "3-class acc", round(acc, 3), "mean prob mass on the 3 words", round(mass, 3), "top token e.g.", r[0]["top"])
PY
            echo F_SMOKE_OK ;;
esac
