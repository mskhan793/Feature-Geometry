#!/bin/bash
# Five-fold summary of the PCA/FDA-GBC runs of both datasets (task.md eeb9707), read-only, one small CPU task.
#
# Usage: sbatch [--dependency=afterok:ARRAY] pca_fda_gbc_summary.sh CODE_DIR ROOT
#SBATCH --job-name=fg_pcafda_sum
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail  # no -u: the module scripts read unset variables

export CODE=$1 ROOT=$2
echo "start $(date -Is) job $SLURM_JOB_ID"
source "$CODE/scripts/lumi/container_env.sh"
singularity exec "$SIF" bash -c "export PYTHONPATH=$CODE; cd $CODE; python scripts/summarise_pca_fda_gbc.py \
    --dataset FIN-Benthic2 $ROOT/finbenthic2 --dataset AquaMonitor $ROOT/aquamonitor --out $ROOT/summary \
    > $ROOT/summary.log 2>&1"
echo "end $(date -Is)"
