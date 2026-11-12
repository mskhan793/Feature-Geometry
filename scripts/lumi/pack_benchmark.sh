#!/bin/bash
# Pack the train + val images of one benchmark dataset into a byte-verified HDF5 on /flash
# (scripts/benchmark/pack_archive_hdf5.py --splits train val); official test images are not packed.
# Usage: sbatch pack_benchmark.sh CODE_DIR MANIFEST OUT_H5 OUT_MANIFEST CONTENT|none ARCHIVE...
#SBATCH --job-name=fg_pack
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
CODE=$1 MANIFEST=$2 OUT_H5=$3 OUT_MANIFEST=$4 CONTENT=$5; shift 5
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST code $(cat "$CODE/CODE_COMMIT") manifest $MANIFEST"
source "$CODE/scripts/lumi/container_env.sh"
mkdir -p "$(dirname "$OUT_H5")"
C=""; [ "$CONTENT" != none ] && C="--content $CONTENT"
singularity exec "$SIF" bash -c "export PYTHONPATH=$CODE PYTHONUNBUFFERED=1; cd $CODE; python scripts/benchmark/pack_archive_hdf5.py \
    --manifest $MANIFEST --archives $* $C --splits train val --out $OUT_H5 --out-manifest $OUT_MANIFEST"
echo "end $(date -Is)"
