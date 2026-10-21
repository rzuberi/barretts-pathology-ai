#!/bin/bash
# Explore/pert stages. Usage: bash full_x.sh STAGE PROMPT. Never edit while jobs run.
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 30000))
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl/explore/pert; C0=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl
X=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_explore/pert; R=$X/runs; mkdir -p $R $X/preds
BASE=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
export PROMPT=$2
# select best checkpoint of a run by three-class accuracy on val (Task F rule); writes preds/selected_<prefix>.json, prints the choice
select_ck() { python - "$1" "$2" <<'PY'
import json, os, sys
X = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_explore/pert"; run, prefix = sys.argv[1], sys.argv[2]; sc = {}
for d in sorted(os.listdir(run)):
    f = f"{X}/preds/{prefix}__{d}__val.jsonl"
    if d.startswith("checkpoint-") and os.path.exists(f):
        r = [json.loads(l) for l in open(f)]; sc[d] = sum(max(("up", "down", "none"), key=lambda w: x["p_" + w]) == x["label"] for x in r) / len(r)
best = max(sc, key=sc.get); json.dump({"prefix": prefix, "set": "val", "scores": sc, "selected": best}, open(f"{X}/preds/selected_{prefix}.json", "w"), indent=1); print(best)
PY
}
cks() { ls -d $1/checkpoint-* | sort -t- -k2 -n | tr '\n' ',' | sed 's/,$//'; }
case $1 in
  x_zs)    BASE=$BASE TAG=zs_$2 SETS=val,test python $C/xpert_eval.py ;;
  x_pipe)  # SFT -> val-select -> merge -> test ; GRPO -> val-select -> test. Every step skips work already on disk.
           [ -f $R/sft_$2/train_summary.json ] || MODE=sft BASE=$BASE OUT=$R/sft_$2 DATA=$X/data/train_all.jsonl STEPS=${STEPS:-2000} python $C/xpert_train.py
           BASE=$BASE ADAPTERS=$(cks $R/sft_$2) TAG=sft_$2 SETS=val python $C/xpert_eval.py
           S=$(select_ck $R/sft_$2 sft_$2); echo "SFT selected $S"
           [ -f $R/sft_$2_merged/config.json ] || python $C0/merge.py $BASE $R/sft_$2/$S $R/sft_$2_merged
           BASE=$BASE ADAPTERS=$R/sft_$2/$S TAG=sft_$2 SETS=test python $C/xpert_eval.py
           [ -f $R/grpo_$2/train_summary.json ] || MODE=grpo BASE=$R/sft_$2_merged OUT=$R/grpo_$2 DATA=$X/data/grpo_all.jsonl STEPS=300 python $C/xpert_train.py
           BASE=$R/sft_$2_merged ADAPTERS=$(cks $R/grpo_$2) TAG=grpo_$2 SETS=val python $C/xpert_eval.py
           G=$(select_ck $R/grpo_$2 grpo_$2); echo "GRPO selected $G"
           BASE=$R/sft_$2_merged ADAPTERS=$R/grpo_$2/$G TAG=grpo_$2 SETS=test python $C/xpert_eval.py
           echo X_PIPE_OK ;;
  x_grpo)  # short GRPO (resumes from any checkpoint in runs/grpo_<P>), final checkpoint only, val + test
           [ -f $R/grpo_$2/train_summary.json ] || MODE=grpo BASE=$R/sft_$2_merged OUT=$R/grpo_$2 DATA=$X/data/grpo_all.jsonl STEPS=${STEPS:-150} python $C/xpert_train.py
           BASE=$R/sft_$2_merged ADAPTERS=$R/grpo_$2/final TAG=grpo_$2 SETS=val,test python $C/xpert_eval.py
           echo X_GRPO_OK ;;
  x_think) # reasoning mode: GRPO from base with thinking on (capped), K prompt; eval = trace then answer-word probabilities
           [ -f $R/think_$2/train_summary.json ] || MODE=grpo THINK=1 MAXCOMP=${MAXCOMP:-256} BASE=$BASE OUT=$R/think_$2 DATA=$X/data/grpo_all.jsonl STEPS=${STEPS:-300} python $C/xpert_train.py
           THINK=1 MAXTHINK=${MAXCOMP:-256} MAXMODEL=1536 BASE=$BASE ADAPTERS=,$R/think_$2/final TAG=think_$2 SETS=val N=1500 python $C/xpert_eval.py
           THINK=1 MAXTHINK=${MAXCOMP:-256} MAXMODEL=1536 BASE=$BASE ADAPTERS=$R/think_$2/final TAG=think_$2 SETS=test python $C/xpert_eval.py
           echo X_THINK_OK ;;
  x_smoke) SM=$X/smoke_$2; rm -rf $SM; mkdir -p $SM
           BASE=$BASE TAG=smk_zs_$2 SETS=val N=200 python $C/xpert_eval.py 2>&1 | grep -E "items in|Error" | tail -2
           MODE=sft BASE=$BASE OUT=$SM/sft DATA=$X/data/train_all.jsonl STEPS=10 N=200 python $C/xpert_train.py 2>&1 | grep -E "TRAIN DONE|Error|train_runtime" | tail -3
           python $C0/merge.py $BASE $SM/sft/final $SM/merged 2>&1 | tail -1
           MODE=grpo BASE=$SM/merged OUT=$SM/grpo DATA=$X/data/grpo_all.jsonl STEPS=3 N=64 python $C/xpert_train.py 2>&1 | grep -E "'reward'|TRAIN DONE|Error|train_runtime" | tail -4
           BASE=$SM/merged ADAPTERS=$SM/grpo/final TAG=smk_grpo_$2 SETS=val N=200 python $C/xpert_eval.py 2>&1 | grep -E "items in|Error" | tail -2
           THINK=1 MAXTHINK=64 MAXMODEL=1536 BASE=$BASE TAG=smk_think_$2 SETS=val N=50 python $C/xpert_eval.py 2>&1 | grep -E "items in|Error" | tail -2
           python - <<PY
import json
for t in ("smk_zs_$2", "smk_grpo_$2__final", "smk_think_$2"):
    r = [json.loads(l) for l in open("$X/preds/" + t + "__val.jsonl")]
    acc = sum(max(("up","down","none"), key=lambda w: x["p_" + w]) == x["label"] for x in r) / len(r); mass = sum(x["mass"] for x in r) / len(r)
    print(t, "n", len(r), "3-class acc", round(acc, 3), "mean prob mass", round(mass, 3), "top e.g.", repr(r[0]["top"]), repr(r[0].get("think", ""))[:150])
PY
           echo X_SMOKE_OK ;;
esac
