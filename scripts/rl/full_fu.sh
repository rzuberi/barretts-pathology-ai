#!/bin/bash
# Follow-ups C (seeds) and D (abstention fixes). Never edit while jobs run (bash reads scripts lazily).
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 30000))
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl; R=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_grader; FU=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_followups
BASE=$(ls -d $HF_HOME/hub/models--google--medgemma-4b-it/snapshots/*/ | head -1); P1=$R/runs/p1_best_merged; D=$R/data
case $1 in
  c_sft)     SEED=$2 MODE=sft BASE=$BASE OUT=$FU/C/runs/sft_s$2 STEPS=900 python $C/train.py ;;
  c_merge)   python $C/merge.py $BASE $FU/C/runs/sft_s$2/final $FU/C/runs/sft_s$2_merged ;;
  c_grpo)    SEED=$2 MODE=grpo BASE=$FU/C/runs/sft_s$2_merged OUT=$FU/C/runs/grpo_sft_s$2 STEPS=300 python $C/train.py ;;
  c_evalsft) BASE=$BASE ADAPTER=$FU/C/runs/sft_s$2/final TAG=sft_s$2__final SETS=val,test,test485 PREDS=$FU/C/preds python $C/evaluate.py ;;
  c_ckpt)    BASE=$FU/C/runs/sft_s$2_merged ADAPTER=$FU/C/runs/grpo_sft_s$2/$3 TAG=grpo_sft_s$2__$3 SETS=val PREDS=$FU/C/preds python $C/evaluate.py ;;
  c_best)    BASE=$FU/C/runs/sft_s$2_merged ADAPTER=$FU/C/runs/grpo_sft_s$2/$3 TAG=grpo_sft_s$2__$3 SETS=val,test,test485 PREDS=$FU/C/preds python $C/evaluate.py ;;
  d_train)   RW=juryabs; [ $2 = d2 ] && RW=brier; REWARD=$RW MODE=grpo BASE=$P1 OUT=$FU/D/runs/$2 STEPS=300 EXTRA=$D/all_lowconf.jsonl KEEP_EXTRA=1 python $C/train.py ;;
  d_ckpt)    if [ $2 = d1 ]; then X="UNSURE=1"; else X="CONF=1"; fi; env $X LOGPROBS=1 BASE=$P1 ADAPTER=$FU/D/runs/$2/$3 TAG=$2__$3 SETS=val PREDS=$FU/D/preds python $C/evaluate.py ;;
  d_best)    if [ $2 = d1 ]; then X="UNSURE=1"; else X="CONF=1"; fi; env $X LOGPROBS=1 BASE=$P1 ADAPTER=$FU/D/runs/$2/$3 TAG=$2__$3 SETS=val,pop,test485 PREDS=$FU/D/preds python $C/evaluate.py ;;
  smoke)     SM=$FU/smoke; rm -rf $SM; mkdir -p $SM
             SEED=1 MODE=sft BASE=$BASE OUT=$SM/sft STEPS=5 N_TRAIN=40 python $C/train.py 2>&1 | tail -2
             for v in juryabs brier; do REWARD=$v MODE=grpo BASE=$P1 OUT=$SM/$v STEPS=5 N_TRAIN=40 EXTRA=$D/all_lowconf.jsonl KEEP_EXTRA=1 python $C/train.py 2>&1 | grep -E "reward \w+;|'reward'|TRAIN DONE|Error" | tail -3; done
             CONF=1 LOGPROBS=1 BASE=$P1 ADAPTER=$SM/brier/final TAG=smk_d2 SETS=val N=50 PREDS=$SM/preds python $C/evaluate.py 2>&1 | grep -E "parse rate|Error" | tail -2
             python -c "import json; r=[json.loads(l) for l in open('$SM/preds/smk_d2__val.jsonl')]; print('conf parsed', sum(x.get('conf') is not None for x in r), '/', len(r), 'example', r[0]['raw'][:160])"
             UNSURE=1 LOGPROBS=1 BASE=$P1 ADAPTER=$SM/juryabs/final TAG=smk_d1 SETS=val N=50 PREDS=$SM/preds python $C/evaluate.py 2>&1 | grep -E "parse rate|Error" | tail -2
             echo FU_SMOKE_OK ;;
esac
