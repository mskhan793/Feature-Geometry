#!/bin/bash
# C1 measured pilots (task.md d57a630), submitted from a LUMI login node. Not run automatically.
#  - throughput pilot: each encoder on FGVC-Aircraft train+val (6,667 images, packed HDF5), one standard-g node
#    (8 GCDs) per encoder; extract_encoder.sh also verifies the cache and compares 256 rows CPU vs GPU;
#  - biological sample: each encoder on the fixed 512-image BIOSCAN-5M train sample, twice (repeat reproducibility),
#    on one GCD; the dinov3_l / dinov2_l / bioclip2 caches are then compared with the v1 caches
#    (compare_v1_bioscan_cache.py, CPU follow-up job).
# Usage: submit_c1_pilots.sh CODE_DIR OUT_ROOT
set -euo pipefail
CODE=$1 OUT=$2
F=/flash/project_462001534/feature_geometry/benchmark
X="$CODE/scripts/lumi/extract_encoder.sh"
mkdir -p "$OUT/logs"
cd "$OUT/logs"
for e in dinov3_l dinov2_l bioclip2 siglip2; do
    sbatch --job-name=fg_c1_air_$e --time=00:45:00 "$X" "$CODE" "$e" "$F/extract/fgvc_aircraft_trainval.parquet" \
        --hdf5 "$F/fgvc_aircraft/fgvc_aircraft_trainval.h5" "$OUT/aircraft/$e"
    for rep in a b; do
        sbatch --job-name=fg_c1_bio_$e$rep --ntasks-per-node=1 --gpus-per-node=1 --mem=60G --time=00:30:00 "$X" "$CODE" "$e" \
            "$F/extract/bioscan5m_train512.parquet" --hdf5 "$F/bioscan5m/bioscan5m_trainval.h5" "$OUT/bioscan512_$rep/$e"
    done
done
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %R"
