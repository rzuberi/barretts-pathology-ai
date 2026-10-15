#!/usr/bin/env bash
# ISBI rerun, CPU phase: slide table (login node) -> 32 pack shards + 32 QC shards on epyc -> finalizer (afterok).
set -uo pipefail
A=/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun; C=$A/code; mkdir -p $A/slurm
source ~/miniforge3/etc/profile.d/conda.sh; conda activate pathology
python $C/make_slide_table.py || exit 1
NS=32; ids=()
for k in $(seq 0 $((NS-1))); do
  ids+=($(sbatch --parsable --partition=epyc --time=01:00:00 --cpus-per-task=2 --mem=16G --job-name=isbi_pack_$k --output=$A/slurm/pack_%j.out --error=$A/slurm/pack_%j.err --export=ALL,SHARD=$k,N_SHARDS=$NS --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && python $C/pack_bags.py"))
  ids+=($(sbatch --parsable --partition=epyc --time=02:00:00 --cpus-per-task=2 --mem=8G --job-name=isbi_qc_$k --output=$A/slurm/qc_%j.out --error=$A/slurm/qc_%j.err --export=ALL,SHARD=$k,N_SHARDS=$NS --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate erin && python $C/qc_all.py"))
done
dep=$(IFS=:; echo "${ids[*]}")
FIN=$(sbatch --parsable --partition=epyc --time=00:20:00 --cpus-per-task=2 --mem=8G --job-name=isbi_finalize --dependency=afterok:$dep --output=$A/slurm/finalize_%j.out --error=$A/slurm/finalize_%j.err --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && python $C/finalize_pack_qc.py")
echo "submitted ${#ids[@]} shard jobs + finalizer $FIN"
