#!/usr/bin/env bash
# Campaign submitter: every small task is raced on all partitions AND a few pull workers per partition drain the same task list.
# Whoever starts first takes a task (feasibility/run_task.sh: done marker + atomic mkdir lock + squeue stale-lock check); duplicates exit in
# seconds. When the last task is done, whoever notices cancels every still-pending job of the campaign (scripts/cluster/campaign_cleanup.sh).
#
# usage: bash scripts/cluster/campaign.sh PREFIX TASKS.tsv [options]
#   TASKS.tsv: one task per line, tab-separated: name<TAB>conda_env<TAB>command   (command runs from the thesis repo root)
#   --partitions "epyc rocm cuda h200"   partitions to use (h200 gets --qos=h200_preempt --exclude=clust1-h200-1)
#   --workers N        pull workers per partition (default 2)
#   --per-task yes|no  also race every task as its own job on every partition (default yes; auto-no if it would exceed --budget)
#   --budget N         max new submissions for this campaign (default 500; the cluster caps a user at about 660 queued jobs)
#   --gres gpu:1                        GPU request added to every job (workers and per-task)
#   --cpus C --mem M --time T          per-job resources (defaults 2, 8G, 2:00:00; workers get --worker-time, default 6:00:00)
#   status: bash scripts/cluster/campaign.sh status PREFIX
set -uo pipefail
TH=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; RUNS=$TH/feasibility/runs; CAMP=$RUNS/_campaigns
if [ "${1:-}" = status ]; then
  P=$2; D=$CAMP/$P; tot=$(grep -c . $D/tasks.tsv); done_=0; lock=0
  while IFS=$'\t' read -r name env cmd; do [ -z "$name" ] && continue; [ -f $RUNS/${P}__$name/done.json ] && done_=$((done_+1)); [ -d $RUNS/${P}__$name/.lock ] && lock=$((lock+1)); done < $D/tasks.tsv
  echo "campaign $P: $done_/$tot done, $lock running"; squeue -u $USER -h -o "%j %P %T" | awk -v p="${P}__" 'index($1,p)==1 {print $2, $3}' | sort | uniq -c; exit 0
fi
P=$1; TASKS=$2; shift 2
PARTS="epyc rocm cuda h200"; W=2; PER=yes; BUDGET=500; CPUS=2; MEM=8G; TIME=2:00:00; WTIME=6:00:00; GRES=""
while [ $# -gt 0 ]; do case $1 in --partitions) PARTS=$2;; --workers) W=$2;; --per-task) PER=$2;; --budget) BUDGET=$2;; --cpus) CPUS=$2;; --mem) MEM=$2;; --time) TIME=$2;; --worker-time) WTIME=$2;; --gres) GRES="--gres=$2";; *) echo "unknown option $1"; exit 2;; esac; shift 2; done
D=$CAMP/$P; mkdir -p $D; cp $TASKS $D/tasks.tsv; echo $P > $D/prefix; echo "$CPUS $MEM $TIME" > $D/resources
N=$(grep -c . $D/tasks.tsv); NP=$(echo $PARTS | wc -w); need=$(( N * NP + W * NP ))
if [ "$PER" = yes ] && [ $need -gt $BUDGET ]; then echo "per-task racing would need $need submissions (> budget $BUDGET): workers only"; PER=no; fi
extra_for() { [ "$1" = h200 ] && echo "--qos=h200_preempt --exclude=clust1-h200-1"; }
sub=0
for part in $PARTS; do
  for i in $(seq 1 $W); do
    sbatch -p $part $(extra_for $part) $GRES -t $WTIME -c $CPUS --mem=$MEM -J ${P}__worker -o $D/worker_%j.log --wrap="cd $TH && bash scripts/cluster/pull_worker.sh $D" > /dev/null && sub=$((sub+1))
  done
done
if [ "$PER" = yes ]; then
  while IFS=$'\t' read -r name env cmd; do
    [ -z "$name" ] && continue
    for part in $PARTS; do
      sbatch -p $part $(extra_for $part) $GRES -t $TIME -c $CPUS --mem=$MEM -J ${P}__$name -o $D/task_%x_%j.log \
        --wrap="cd $TH && TASK_NAME=${P}__$name CONDA_ENV='$env' TASK_CMD='$cmd' bash feasibility/run_task.sh; bash scripts/cluster/campaign_cleanup.sh $D" > /dev/null && sub=$((sub+1))
    done
  done < $D/tasks.tsv
fi
echo "campaign $P: $N tasks, $sub jobs submitted on [$PARTS] (workers per partition $W, per-task racing $PER)"
