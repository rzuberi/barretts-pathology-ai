#!/bin/bash
A=/mnt/scratche/slow/fmlab/zuberi01/analysis/tilebudget; ME=$$
for p in $(pgrep -u zuberi01 -f "bash $A/code/tb_chain.sh"); do [ "$p" != "$ME" ] && [ "$(ps -o comm= -p $p)" = bash ] && kill $p && echo "killed old chain $p"; done
sleep 1; echo "$(date +%H:%M) chain restarted after --partitions quoting fix" >> $A/chain.log
setsid nohup bash $A/code/tb_chain.sh > /dev/null 2>&1 < /dev/null &
sleep 20; tail -3 $A/chain.log; squeue -h -u zuberi01 -o "%P %T %M %j" | grep -E "tbsel|tbsmoke"
