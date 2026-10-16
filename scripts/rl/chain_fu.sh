#!/bin/bash
# Cluster-side chain for follow-ups C, D, E, F. Each part starts only after its smoke test has a done marker. Log: rl_followups/chain.log
T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; C=$T/scripts/rl; RUNS=$T/feasibility/runs; FU=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_followups
SQ=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql; L=$FU/chain.log; source ~/miniforge3/etc/profile.d/conda.sh; conda activate pathology
log() { echo "$(date +%H:%M) $*" >> $L; }; done_() { [ -f $RUNS/$1/done.json ]; }; waitfor() { while ! done_ $1; do sleep 180; done; }
race() { done_ $1 || bash $C/race3.sh "$@" >/dev/null; }; ckpts() { ls -d $1/checkpoint-* 2>/dev/null | xargs -n1 basename | sort -t- -k2 -n; }
FUS=$C/full_fu.sh; FE=$C/full_e.sh; FF=$C/full_f.sh; log "chain start"
# ---------- C: seeds 1, 2
for s in 1 2; do ( waitfor fu_smoke; race fu_c_sft_$s "c_sft $s" 3:00:00 $FUS; waitfor fu_c_sft_$s; log "C sft seed $s done"
  race fu_c_evalsft_$s "c_evalsft $s" 2:00:00 $FUS; race fu_c_merge_$s "c_merge $s" 2:00:00 $FUS; waitfor fu_c_merge_$s
  race fu_c_grpo_$s "c_grpo $s" 14:00:00 $FUS; waitfor fu_c_grpo_$s; log "C grpo seed $s done"
  for c in $(ckpts $FU/C/runs/grpo_sft_s$s); do race fu_c_ck_${s}_$c "c_ckpt $s $c" 2:00:00 $FUS; done; for c in $(ckpts $FU/C/runs/grpo_sft_s$s); do waitfor fu_c_ck_${s}_$c; done
  b=$(python $C/sel.py acc $FU/C/preds grpo_sft_s$s val); log "C seed $s best $b"; race fu_c_best_$s "c_best $s $b" 3:00:00 $FUS; waitfor fu_c_best_$s; waitfor fu_c_evalsft_$s; log "C SEED $s DONE" ) & done
# ---------- D: abstention fixes
for v in d1 d2; do ( waitfor fu_smoke; race fu_d_train_$v "d_train $v" 14:00:00 $FUS; waitfor fu_d_train_$v; log "D $v trained"
  for c in $(ckpts $FU/D/runs/$v); do race fu_d_ck_${v}_$c "d_ckpt $v $c" 2:00:00 $FUS; done; for c in $(ckpts $FU/D/runs/$v); do waitfor fu_d_ck_${v}_$c; done
  b=$(SEL_DIR=$FU/D python $C/select_best.py p2 $v); log "D $v best $b"; race fu_d_best_$v "d_best $v $b" 4:00:00 $FUS; waitfor fu_d_best_$v; log "D $v DONE" ) & done
# ---------- E: agent
( waitfor e_smoke; race fu_e_sft e_sft 2:00:00 $FE; waitfor fu_e_sft; log "E sft done"
  for c in $(ckpts $SQ/agent_runs/sft) final; do race fu_e_sftck_$c "e_sftck $c" 1:00:00 $FE; done; for c in $(ckpts $SQ/agent_runs/sft) final; do waitfor fu_e_sftck_$c; done
  b=$(python $C/sel.py sql $SQ/agsplit_preds agsplit_sft val); log "E sft best $b"
  race fu_e_merge "e_merge $b" 1:00:00 $FE; race fu_e_a "e_a $b" 1:00:00 $FE; waitfor fu_e_merge
  race fu_e_b e_b 2:00:00 $FE; race fu_e_grpo e_grpo 16:00:00 $FE; waitfor fu_e_grpo; log "E grpo done"
  for c in $(ckpts $SQ/agent_runs/grpo); do race fu_e_ck_$c "e_ck $c" 2:00:00 $FE; done; for c in $(ckpts $SQ/agent_runs/grpo); do waitfor fu_e_ck_$c; done
  b=$(python $C/sel.py sql $SQ/agent_preds agent_grpo val); log "E grpo best $b"; race fu_e_best "e_best $b" 2:00:00 $FE; waitfor fu_e_best; waitfor fu_e_a; waitfor fu_e_b; log "E DONE" ) &
# ---------- F: perturbation
( waitfor f_smoke; race fu_f_zs f_zs 3:00:00 $FF
  for V in all nor; do ( race fu_f_sft_$V "f_sft $V" 8:00:00 $FF; waitfor fu_f_sft_$V; log "F sft $V done"
    for c in $(ckpts $FU/F/runs/sft_$V); do race fu_f_sftck_${V}_$c "f_sftck $V $c" 1:00:00 $FF; done; for c in $(ckpts $FU/F/runs/sft_$V); do waitfor fu_f_sftck_${V}_$c; done
    VS=val; [ $V = nor ] && VS=val_nor; b=$(python $C/sel.py pert $FU/F/preds sft_$V $VS); log "F sft $V best $b"
    race fu_f_merge_$V "f_merge $V $b" 1:00:00 $FF; race fu_f_testsft_$V "f_testsft $V $b" 3:00:00 $FF; waitfor fu_f_merge_$V
    race fu_f_grpo_$V "f_grpo $V" 10:00:00 $FF; waitfor fu_f_grpo_$V; log "F grpo $V done"
    for c in $(ckpts $FU/F/runs/grpo_$V); do race fu_f_grpock_${V}_$c "f_grpock $V $c" 1:00:00 $FF; done; for c in $(ckpts $FU/F/runs/grpo_$V); do waitfor fu_f_grpock_${V}_$c; done
    b=$(python $C/sel.py pert $FU/F/preds grpo_$V $VS); log "F grpo $V best $b"; race fu_f_testgrpo_$V "f_testgrpo $V $b" 3:00:00 $FF; waitfor fu_f_testgrpo_$V; waitfor fu_f_testsft_$V; log "F $V DONE" ) & done
  wait; waitfor fu_f_zs; log "F DONE" ) &
wait; log "CHAIN DONE"
