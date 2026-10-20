#!/bin/bash
# restart the chain: cancel smoke jobs, clear their task dirs, keep the old log, start op_chain.sh detached
A=/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia
for j in $(squeue -u $USER -h -o "%i %j" | awk '$2 ~ /^op_trsmoke__/ {print $1}'); do scancel $j; done
sleep 3; rm -rf /mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/runs/op_trsmoke__*
[ -f $A/chain.log ] && mv $A/chain.log $A/chain_v1.log
setsid nohup bash $A/code/op_chain.sh > $A/chain.nohup 2>&1 < /dev/null &
