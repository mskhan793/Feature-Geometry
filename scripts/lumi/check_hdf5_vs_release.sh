#!/bin/bash
# Compare every image of the downloaded AquaMonitor release (111 WebDataset shards) with our self-built HDF5, by name
# and bytes, one worker per shard (scripts/compare_hdf5_with_release_shard.py, read-only). Exits non-zero unless every
# released image is byte-identical and every HDF5 position is covered exactly once.
#
# Usage: sbatch check_hdf5_vs_release.sh CODE_DIR RELEASE_DIR HDF5 OUT_JSON
#SBATCH --job-name=fg_hdf5_vs_release
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

CODE=$1 RELEASE=$2 HDF5=$3 OUT=$4
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260731_122833/lumi-multitorch-full-u24r70f21m50t210-20260731_122833.sif
echo "start $(date -Is) shards $(ls "$RELEASE"/images/*.tar | wc -l) hdf5 $HDF5"
singularity exec "$SIF" python "$CODE/scripts/compare_hdf5_with_release_shard.py" --hdf5 "$HDF5" --out "$OUT" \
    --workers "$SLURM_CPUS_PER_TASK" "$RELEASE"/images/*.tar
singularity exec "$SIF" python -c "
import json, sys
r = json.load(open(sys.argv[1])); t, c = r['totals'], r['coverage']
ok = (t['members'] == t['bytes_identical'] == c['hdf5_rows'] == c['rows_seen_once'] and t['not_in_hdf5'] == 0)
print('PASS' if ok else 'FAIL', t, c)
sys.exit(0 if ok else 1)" "$OUT"
echo "end $(date -Is)"
