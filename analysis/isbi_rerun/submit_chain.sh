#!/usr/bin/env bash
# Resubmit: 32 pack shards (contiguous chunks, 2h) -> finalizer -> 5 fold jobs (twinned) -> part3 (twinned)
set -uo pipefail
A=/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun; C=$A/code; T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; F=$T/feasibility
rm -f $A/pack/bags_fp16.dat $A/pack/meta_shard*.json $A/pack/chunk_*.npy*
NS=32; ids=()
for k in $(seq 0 $((NS-1))); do ids+=($(sbatch --parsable --partition=epyc --time=02:00:00 --cpus-per-task=4 --mem=24G --job-name=isbi_pack_$k --output=$A/slurm/pack2_%j.out --error=$A/slurm/pack2_%j.err --export=ALL,SHARD=$k,N_SHARDS=$NS --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && python $C/pack_bags.py")); done
dep=$(IFS=:; echo "${ids[*]}")
FIN=$(sbatch --parsable --partition=epyc --time=00:20:00 --cpus-per-task=2 --mem=8G --job-name=isbi_finalize --dependency=afterok:$dep --output=$A/slurm/finalize_%j.out --error=$A/slurm/finalize_%j.err --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate pathology && python $C/finalize_pack_qc.py")
fids=()
for f in 0 1 2 3 4; do name=isbi_fold$f; rm -rf $F/runs/$name
  for part in h200 cuda; do extra=""; [[ $part == h200 ]] && extra="--qos=h200_preempt"
    fids+=($(sbatch --parsable --job-name=$name --partition=$part $extra --dependency=afterok:$FIN --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 --output=$A/slurm/${name}_${part}_%j.out --error=$A/slurm/${name}_${part}_%j.err --export=ALL,TASK_NAME=$name,CONDA_ENV=virchow2,FOLD=$f,TASK_CMD="python $C/train_fold.py" --wrap="bash $F/run_task.sh")); done; done
fdep=$(IFS=:; echo "${fids[*]}"); rm -rf $F/runs/isbi_part3
for part in h200 cuda; do extra=""; [[ $part == h200 ]] && extra="--qos=h200_preempt"
  sbatch --parsable --job-name=isbi_part3 --partition=$part $extra --dependency=afterok:$fdep --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 --output=$A/slurm/part3_${part}_%j.out --error=$A/slurm/part3_${part}_%j.err --export=ALL,TASK_NAME=isbi_part3,CONDA_ENV=virchow2,TASK_CMD="python $C/part3_swg.py" --wrap="bash $F/run_task.sh" >/dev/null; done
echo "pack ${#ids[@]} jobs -> finalizer $FIN -> folds ${#fids[@]} jobs -> part3 x2"
