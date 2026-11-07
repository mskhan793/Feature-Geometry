#!/bin/bash
# Three-baseline LUMI smoke test (task.md 8ac12a7): one GCD per encoder (BioCLIP-2, DINOv2-L, DINOv3-L) in parallel,
# each running scripts/lumi/smoke_baselines_task.sh. Infrastructure check only: no full extraction, no evaluation.
#
# Usage: sbatch smoke_baselines.sh CODE_DIR SMOKE_ROOT
#SBATCH --job-name=fg_smoke_baselines
#SBATCH --account=project_462001534
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail  # no -u: the module scripts read unset variables

export CODE=$1
export SMOKE=$2/$SLURM_JOB_ID
source "$CODE/scripts/lumi/container_env.sh"
export V1=/scratch/project_462001534/aqua_v1
export HF_HOME=$V1/scratch_$USER/hf_cache
export HDF5=/flash/project_462001534/data_hdf_aqua/aquamonitor_complete.h5
export MANIFEST=/flash/project_462001534/feature_geometry/manifests/aquamonitor_classif_manifest.parquet
export V1_PARQUET=$V1/data/aquamonitor-monitor-with-hierarchy_suffix_fixed_author2021_2022_ood_full2021trainval_full2022test_detectall.parquet.gzip
export ROWS=16384 CHUNK=4096
mkdir -p "$SMOKE"
for f in "$CODE/scripts/extract_features.py" "$HDF5" "$MANIFEST" "$V1_PARQUET"; do  # fail fast if anything is not visible
    singularity exec "$SIF" test -r "$f" || { echo "not readable in the container: $f" >&2; exit 1; }
done
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST code $(cat "$CODE/CODE_COMMIT") smoke $SMOKE"
srun --kill-on-bad-exit=0 bash "$CODE/scripts/lumi/smoke_baselines_task.sh"
echo "end $(date -Is)"
