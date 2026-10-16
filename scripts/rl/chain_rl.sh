#!/bin/bash
# Cluster-side chain for projects 1 and 3 after stage 1. Polls done markers, submits raced tasks, logs to analysis/rl_grader/chain.log.
T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; C=$T/scripts/rl; RUNS=$T/feasibility/runs; L=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_grader/chain.log
P1=$C/full_p1.sh; P3=$C/full_p3.sh; source ~/miniforge3/etc/profile.d/conda.sh; conda activate pathology
log() { echo "$(date +%H:%M) $*" >> $L; }
done_() { [ -f $RUNS/$1/done.json ]; }
waitfor() { while ! done_ $1; do sleep 120; done; }
race() { done_ $1 || bash $C/race3.sh "$@" >/dev/null; }
ckpts() { ls -d $1/checkpoint-* 2>/dev/null | xargs -n1 basename | sort -t- -k2 -n; }
log "chain start"
# --- project 1: SFT -> eval + merge -> GRPO-from-SFT
( waitfor p1s1__sft; log "p1 sft done"; race p1_eval_sft "eval_best sft final" 3:00:00; race p1_merge_sft merge_sft 2:00:00
  waitfor p1_merge_sft; log "p1 sft merged"; race p1_grpo_sft grpo_sft 12:00:00; waitfor p1_grpo_sft; log "p1 grpo_sft done"
  for c in $(ckpts /mnt/scratche/slow/fmlab/zuberi01/analysis/rl_grader/runs/grpo_sft); do race p1_ck_grpo_sft_$c "eval_ckpt grpo_sft $c" 2:00:00; done
  for c in $(ckpts /mnt/scratche/slow/fmlab/zuberi01/analysis/rl_grader/runs/grpo_sft); do waitfor p1_ck_grpo_sft_$c; done
  b=$(python $C/select_best.py p1 grpo_sft); log "p1 grpo_sft best $b"; race p1_best_grpo_sft "eval_best grpo_sft $b" 3:00:00; waitfor p1_best_grpo_sft; log "P1 GRPO_SFT EVAL DONE" ) &
# --- project 1: GRPO-from-base -> checkpoint evals -> best
( waitfor p1_grpo_base; log "p1 grpo_base done"
  for c in $(ckpts /mnt/scratche/slow/fmlab/zuberi01/analysis/rl_grader/runs/grpo_base); do race p1_ck_grpo_base_$c "eval_ckpt grpo_base $c" 2:00:00; done
  for c in $(ckpts /mnt/scratche/slow/fmlab/zuberi01/analysis/rl_grader/runs/grpo_base); do waitfor p1_ck_grpo_base_$c; done
  b=$(python $C/select_best.py p1 grpo_base); log "p1 grpo_base best $b"; race p1_best_grpo_base "eval_best grpo_base $b" 3:00:00; waitfor p1_best_grpo_base; log "P1 GRPO_BASE EVAL DONE" ) &
# --- project 3: GRPO -> checkpoint evals -> best
( waitfor p3s1__grpo; log "p3 grpo done"
  for c in $(ckpts /mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql/runs/grpo); do race p3_ck_grpo_$c "ckpt grpo $c" 1:00:00 $P3; done
  for c in $(ckpts /mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql/runs/grpo); do waitfor p3_ck_grpo_$c; done
  b=$(python $C/select_best.py p3 grpo); log "p3 grpo best $b"; race p3_best_grpo "best grpo $b" 1:00:00 $P3; waitfor p3_best_grpo; log "P3 GRPO EVAL DONE" ) &
wait; log "CHAIN DONE"
