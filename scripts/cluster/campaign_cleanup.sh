#!/usr/bin/env bash
# If every task of the campaign has a done marker, cancel all of the campaign's still-PENDING jobs (duplicates and idle workers).
D=$1; RUNS=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/runs; P=$(cat $D/prefix)
while IFS=$'\t' read -r name env cmd; do [ -z "$name" ] && continue; [ -f $RUNS/${P}__$name/done.json ] || exit 0; done < $D/tasks.tsv
ids=$(squeue -u $USER -h -t PD -o "%i %j" | awk -v p="${P}__" 'index($2,p)==1 {print $1}')
[ -n "$ids" ] && echo $ids | xargs scancel && echo "[cleanup] campaign $P complete: cancelled $(echo $ids | wc -w) pending jobs"
[ -f $D/COMPLETE ] || date -Iseconds > $D/COMPLETE
