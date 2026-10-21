#!/bin/bash
# bash run_metrics.sh TAG1,TAG2 BESTTAG : project-1 style metrics + project-2 triage (tie-aware estimator) at 20% of reports
B=/mnt/scratche/slow/fmlab/zuberi01; C=$B/phd/thesis/scripts/rl/explore/clf; O=$B/analysis/rl_explore/clf
source ~/miniforge3/etc/profile.d/conda.sh; conda activate pathology; cd $C
TAGS=$1 python clf_metrics.py | tee $O/metrics.txt
PDIR=$O ALPHA_TAGS=$1 BEST=$2 PRIMARY_FRAC=0.20 OUT=$O/triage.json python $B/phd/thesis/scripts/rl/p2_metrics.py | tee $O/triage.txt
