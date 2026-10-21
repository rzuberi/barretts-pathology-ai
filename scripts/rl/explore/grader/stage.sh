#!/bin/bash
# Explore/grader stages (race-to-run via race3.sh). Never edit while a job runs it.
# smoke | a1 | a2 | a3 | ckpt RUN CK | best RUN CK
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 30000))
B=/mnt/scratche/slow/fmlab/zuberi01; C=$B/phd/thesis/scripts/rl; G=$C/explore/grader; R=$B/analysis/rl_grader; O=$B/analysis/rl_explore/grader
SFT=$R/runs/sft_merged; D=$R/data; V=$O/votes.json; mkdir -p $O/runs $O/preds; chmod 700 $O
case $1 in
  smoke)  SM=$O/smoke; rm -rf $SM; mkdir -p $SM
          REWARD=soft VOTES=$V BASE=$SFT OUT=$SM/a1 STEPS=20 N_TRAIN=50 EXTRA=$D/all_lowconf.jsonl KEEP_EXTRA=1 python $G/train_soft.py 2>&1 | grep -E "soft reward|grpo:|resuming|'reward'|TRAIN DONE|Error|error" | tail -8
          BASE=$SFT ADAPTER=$SM/a1/final TAG=smk_a1 SETS=val N=50 PREDS=$SM/preds python $C/evaluate.py 2>&1 | grep -E "parse rate|Error|EVAL DONE" | tail -3
          python -c "import json; r=[json.loads(l) for l in open('$SM/preds/smk_a1__val.jsonl')]; print('smoke val acc', sum(x['pred']==x['label'] for x in r)/len(r), 'parsed', sum(x['pred'] is not None for x in r), '/', len(r))"
          echo GRADER_SMOKE_OK ;;
  a1)     REWARD=soft VOTES=$V BASE=$SFT OUT=$O/runs/a1 STEPS=300 EXTRA=$D/all_lowconf.jsonl KEEP_EXTRA=1 python $G/train_soft.py ;;
  a2)     REWARD=soft VOTES=$V BASE=$SFT OUT=$O/runs/a2 STEPS=300 python $G/train_soft.py ;;
  a3)     # continue the seed-0 project-1 GRPO run (P1 reward) from its checkpoint-300 to 600 steps; checkpoint copied once, resume-safe after
          mkdir -p $O/runs/a3; [ -d $O/runs/a3/checkpoint-300 ] || cp -r $R/runs/grpo_sft/checkpoint-300 $O/runs/a3/
          REWARD=p1 BASE=$SFT OUT=$O/runs/a3 STEPS=600 python $G/train_soft.py ;;
  mine)   BASE=$SFT OUT=$O/mined.json python $G/mine.py ;;
  a4)     # P1 reward, P1 data, but only prompts whose 8 SFT samples were mixed (mined once by stage 'mine')
          REWARD=p1 MINED=$O/mined.json BASE=$SFT OUT=$O/runs/a4 STEPS=300 python $G/train_soft.py ;;
  short)  # deadline arm (23:10 20 Oct): 100 GRPO steps, soft reward + low-agreement sections, from the P1 RL model; train + select on val + test in ONE job
          P1=$R/runs/p1_best_merged; RUN=$O/runs/short
          REWARD=soft VOTES=$V BASE=$P1 OUT=$RUN STEPS=100 EXTRA=$D/all_lowconf.jsonl KEEP_EXTRA=1 python $G/train_soft.py
          for ck in checkpoint-50 checkpoint-100; do BASE=$P1 ADAPTER=$RUN/$ck TAG=short__$ck SETS=val PREDS=$O/preds python $C/evaluate.py; done
          BEST=$(python $C/sel.py acc $O/preds short val)
          BASE=$P1 ADAPTER=$RUN/$BEST TAG=short__$BEST SETS=val,test,test485 PREDS=$O/preds python $C/evaluate.py
          echo SHORT_DONE $BEST ;;
  ckpt)   BASE=$SFT ADAPTER=$O/runs/$2/$3 TAG=$2__$3 SETS=val PREDS=$O/preds python $C/evaluate.py ;;
  best)   BASE=$SFT ADAPTER=$O/runs/$2/$3 TAG=$2__$3 SETS=val,test,test485 PREDS=$O/preds python $C/evaluate.py ;;
esac
