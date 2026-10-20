#!/bin/bash
# usage: wait_campaign.sh PREFIX  -> blocks until every task of the campaign has done.json, or no PREFIX__ job is queued (squeue must succeed)
P=$1; R=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/runs; T=$R/_campaigns/$P/tasks.tsv; N=$(grep -c . $T)
while :; do
  d=0; while IFS=$'\t' read -r name env cmd; do [ -f $R/${P}__$name/done.json ] && d=$((d+1)); done < $T
  [ $d -ge $N ] && { echo "ALLDONE $d/$N"; exit 0; }
  if q=$(squeue -u zuberi01 -h -o %j 2>/dev/null); then n=$(echo "$q" | grep -c "^${P}__"); [ $n -eq 0 ] && { echo "NOJOBS $d/$N"; exit 1; }; fi
  sleep 30
done
