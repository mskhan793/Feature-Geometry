#!/bin/bash
# CPU checks of the I-JEPA extraction pilot (task.md 3b897b0 gates), after the two 512-image BIOSCAN-5M repeat
# extractions and the FGVC-Aircraft throughput extraction (sbatch --dependency=afterok):
#  - repeat determinism: the two 512-image caches compared by image name (compare_caches_by_name.py, must be identical);
#  - feature dimension, numerical rank and the final-LayerNorm constraint of the pooled feature, with the stored
#    target-encoder norm weights (audit_feature_null_space.py, key prefix ``norm``) on the 512 and Aircraft caches.
# The Aircraft cache's own verification and 256-row CPU-vs-GPU comparison run in check_extraction.sh.
# Usage: sbatch ijepa_pilot_checks.sh CODE_DIR OUT_ROOT
#SBATCH --job-name=fg_ijepa_chk
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
CODE=$1 OUT=$2
echo "start $(date -Is) job $SLURM_JOB_ID code $(cat "$CODE/CODE_COMMIT")"
source "$CODE/scripts/lumi/container_env.sh"
F=/flash/project_462001534/feature_geometry/benchmark
W=/scratch/project_462001534/feature_geometry/weights/ijepa/IN1K-vit.h.14-300e_target_encoder.safetensors
M=$F/extract/bioscan5m_train512.parquet
run() { singularity exec "$SIF" bash -c "export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8; cd $CODE; $*"; }
run python scripts/compare_caches_by_name.py --names $M --a $M=$OUT/bioscan512_a/features_gpu/ijepa_h14 \
    --b $M=$OUT/bioscan512_b/features_gpu/ijepa_h14 --out $OUT/repeat_comparison.json
run python scripts/audit_feature_null_space.py --cache $OUT/bioscan512_a/features_gpu/ijepa_h14 --weights $W \
    --key-prefix norm --out $OUT/null_space_bioscan512.json
run python scripts/audit_feature_null_space.py --cache $OUT/aircraft/features_gpu/ijepa_h14 --weights $W \
    --key-prefix norm --out $OUT/null_space_aircraft.json
echo "end $(date -Is)"
