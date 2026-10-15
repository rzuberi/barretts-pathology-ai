#!/usr/bin/env bash
# Pull worker for scripts/cluster/campaign.sh: walks the campaign's task list (in a job-specific shuffled order to spread contention) and runs
# every task that is neither done nor held by a live job, through feasibility/run_task.sh. Two passes (the second retries failures once).
# Stops early if a whole pass makes no progress (failure floor), then runs the cleanup.
D=$1; TH=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; RUNS=$TH/feasibility/runs; P=$(cat $D/prefix); cd $TH
count_done() { local c=0; while IFS=$'\t' read -r name env cmd; do [ -n "$name" ] && [ -f $RUNS/${P}__$name/done.json ] && c=$((c+1)); done < $D/tasks.tsv; echo $c; }
for pass in 1 2; do
  before=$(count_done)
  while IFS=$'\t' read -r name env cmd; do
    [ -z "$name" ] && continue; [ -f $RUNS/${P}__$name/done.json ] && continue
    ( TASK_NAME=${P}__$name CONDA_ENV="$env" TASK_CMD="$cmd" bash feasibility/run_task.sh ) || echo "[worker ${SLURM_JOB_ID:-}] task $name failed (pass $pass)"
  done < <(shuf --random-source=<(yes ${SLURM_JOB_ID:-0}) $D/tasks.tsv)
  after=$(count_done); echo "[worker ${SLURM_JOB_ID:-}] pass $pass: done $before -> $after of $(grep -c . $D/tasks.tsv)"
  [ "$after" -eq "$before" ] && [ $pass -gt 1 ] && break
done
bash scripts/cluster/campaign_cleanup.sh $D
