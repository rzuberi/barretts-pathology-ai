#!/bin/bash
# Explore/agent stages (expert iteration, reward shaping + schema prompt, training-data augmentation). Never edit while jobs run.
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false SCHEMA_PROMPT=1 SHAPED=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 30000))
B=/mnt/scratche/slow/fmlab/zuberi01; C=$B/phd/thesis/scripts/rl; C2=$C/explore/agent; O=$B/analysis/rl_sql; O2=$B/analysis/rl_explore/agent
mkdir -p $O2/runs $O2/preds $O2/traj; chmod 700 $O2
M=$O/agent_runs/sft_merged; GE=$O/agent_runs/grpo/checkpoint-150; QF=$O/questions_agent.jsonl; QA=$O2/questions_aug.jsonl; EIM=$O2/runs/ei_merged
ckevals() { ADS=$(for d in $O2/runs/$1/checkpoint-*; do echo -n "$(basename $d)=$d,"; done); BASE=$2 ADAPTERS=$ADS TAG=$1 SETS=val python $C2/eval2.py; python $C2/select2.py $O2/preds $1; }
testeval() { CK=$(python -c "import json;print(json.load(open('$O2/preds/selected_$1.json'))['selected'])"); BASE=$2 ADAPTERS="$CK=$O2/runs/$1/$CK" TAG=$1 SETS=test python $C2/eval2.py; }
case $1 in
  baselines) BASE=$M ADAPTERS="taskE_ck150=$GE" TAG=schema SETS=val,test python $C2/eval2.py
             BASE=$M TAG=schema__untrained SETS=val,test python $C2/eval2.py ;;
  jobA)      bash $0 baselines
             BASE=$M OUT=$O2/runs/A STEPS=150 python $C2/grpo2.py; ckevals A $M; testeval A $M ;;
  jobB)      BASE=$M ADAPTERS="taskE=$GE,none=" K=${K:-8} OUT=$O2/traj/train.jsonl python $C2/gen_traj.py
             python $C2/build_sft.py $O2/traj/train.jsonl $O2/traj/ei_sft_data.jsonl 2 1 | tee $O2/traj/ei_sft_stats.json
             BASE=$M DATA=$O2/traj/ei_sft_data.jsonl OUT=$O2/runs/ei_sft EPOCHS=2 python $C2/ei_sft.py
             python $C/merge.py $M $O2/runs/ei_sft/final $EIM
             BASE=$EIM TAG=B_ei_sft SETS=val,test python $C2/eval2.py
             BASE=$EIM OUT=$O2/runs/B STEPS=150 python $C2/grpo2.py; ckevals B $EIM; testeval B $EIM ;;
  jobC)      python $C2/augment.py; python $C2/paraphrase2.py
             QFILE=$QA BASE=$M OUT=$O2/runs/C STEPS=150 python $C2/grpo2.py; ckevals C $M; testeval C $M ;;
  jobD)      [ -f $EIM/config.json ] && [ -f $QA ] || { echo "D needs ei_merged and questions_aug"; exit 1; }
             QFILE=$QA BASE=$EIM OUT=$O2/runs/D STEPS=150 python $C2/grpo2.py; ckevals D $EIM; testeval D $EIM ;;
  smoke)     SM=$O2/smoke; rm -rf $SM; mkdir -p $SM
             BASE=$M ADAPTERS="taskE=$GE,none=" K=3 N=10 OUT=$SM/traj.jsonl python $C2/gen_traj.py 2>&1 | grep -E "episodes|GEN DONE|Error" | tail -3
             python $C2/build_sft.py $SM/traj.jsonl $SM/sft.jsonl 2 1 | tail -3
             BASE=$M DATA=$SM/sft.jsonl OUT=$SM/ei EPOCHS=1 ACC=2 python $C2/ei_sft.py 2>&1 | grep -E "trajectories|TRAIN DONE|Error" | tail -3
             python $C/merge.py $M $SM/ei/final $SM/ei_merged 2>&1 | tail -1
             BASE=$SM/ei_merged OUT=$SM/grpo STEPS=2 B=2 G=4 N_TRAIN=16 python $C2/grpo2.py 2>&1 | grep -E "'step'|TRAIN DONE|Error|error" | tail -4
             BASE=$SM/ei_merged ADAPTERS="checkpoint-2=$SM/grpo/checkpoint-2,taskE=$GE" TAG=smk SETS=val N=12 PREDS=$SM/preds python $C2/eval2.py 2>&1 | grep -E "exec_acc|Error" | tail -3
             python $C2/select2.py $SM/preds smk
             python $C2/augment.py; N=6 OUTQ=$SM/questions_aug.jsonl python $C2/paraphrase2.py 2>&1 | grep -E "paraphrases|Error" | tail -2
             echo X_AGENT_SMOKE_OK ;;
esac
