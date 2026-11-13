#!/bin/bash
# I-JEPA ViT-H/14 extraction pilot (task.md 3b897b0 gates), submitted from a LUMI login node, as the C1 pilots of the
# other encoders (submit_c1_pilots.sh):
#  - throughput: FGVC-Aircraft train+val (6,667 images, packed HDF5) on one standard-g node (8 GCDs), then
#    check_extraction.sh (cache verification, 256-row CPU vs GPU);
#  - determinism: the fixed 512-image BIOSCAN-5M train sample twice, one GCD each;
#  - ijepa_pilot_checks.sh after all three: repeat comparison, dimension / rank / LayerNorm constraint.
# Usage: submit_ijepa_pilot.sh CODE_DIR OUT_ROOT
set -euo pipefail
CODE=$1 OUT=$2
F=/flash/project_462001534/feature_geometry/benchmark
X="$CODE/scripts/lumi/extract_encoder.sh"
mkdir -p "$OUT/logs"
cd "$OUT/logs"
AIR=(ijepa_h14 "$F/extract/fgvc_aircraft_trainval.parquet" --hdf5 "$F/fgvc_aircraft/fgvc_aircraft_trainval.h5" "$OUT/aircraft")
a=$(sbatch --parsable --job-name=fg_ijepa_air --time=00:45:00 "$X" "$CODE" "${AIR[@]}")
c=$(sbatch --parsable --job-name=fg_ijepa_airchk --time=01:00:00 --dependency=afterok:$a --kill-on-invalid-dep=yes \
    "$CODE/scripts/lumi/check_extraction.sh" "$CODE" "${AIR[@]}")
ids=()
for rep in a b; do
    ids+=($(sbatch --parsable --job-name=fg_ijepa_bio$rep --ntasks-per-node=1 --gpus-per-node=1 --mem=60G --time=00:30:00 \
        "$X" "$CODE" ijepa_h14 "$F/extract/bioscan5m_train512.parquet" --hdf5 "$F/bioscan5m/bioscan5m_trainval.h5" \
        "$OUT/bioscan512_$rep"))
done
k=$(sbatch --parsable --dependency=afterok:$a:${ids[0]}:${ids[1]} --kill-on-invalid-dep=yes \
    "$CODE/scripts/lumi/ijepa_pilot_checks.sh" "$CODE" "$OUT")
echo -e "aircraft\t$a\naircraft_check\t$c\nbioscan512_a\t${ids[0]}\nbioscan512_b\t${ids[1]}\nchecks\t$k" | tee "$OUT/logs/submitted.tsv"
