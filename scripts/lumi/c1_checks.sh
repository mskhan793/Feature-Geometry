#!/bin/bash
# C1 follow-up checks (CPU), after submit_c1_pilots.sh: v1 BIOSCAN cache equivalence (dinov3_l, dinov2_l, bioclip2),
# repeat reproducibility of the 512-image sample (run a vs b, every encoder), rank / LayerNorm audit of the Aircraft
# pilot features (with the final-norm weights for the DINO models), and the per-run extraction summary.
# Usage: sbatch c1_checks.sh CODE_DIR C1_ROOT
#SBATCH --job-name=fg_c1_checks
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
CODE=$1 R=$2
echo "start $(date -Is) job $SLURM_JOB_ID code $(cat "$CODE/CODE_COMMIT")"
source "$CODE/scripts/lumi/container_env.sh"
run() { singularity exec "$SIF" bash -c "export PYTHONPATH=$CODE PYTHONUNBUFFERED=1; cd $CODE; $*"; }
S=/flash/project_462001534/feature_geometry/benchmark/extract/bioscan5m_train512.parquet
V1=/scratch/project_462001534/aqua_v1/bioscan
H=/scratch/project_462001534/aqua_v1/scratch_$USER/hf_cache/hub
mkdir -p "$R/checks"
for pair in dinov3_l:dinov3 dinov2_l:dinov2 bioclip2:bioclip2; do
    e=${pair%%:*}; v=${pair##*:}
    run python scripts/benchmark/compare_v1_bioscan_cache.py --v1-index $V1/bioscan_index.parquet --v1-cache $V1/cache_$v \
        --split train --sample $S --new "$S=$R/bioscan512_a/$e/features_gpu/$e" --out "$R/checks/v1_equivalence_$e.json"
done
for e in dinov3_l dinov2_l bioclip2 siglip2; do
    run python scripts/compare_caches_by_name.py --names $S --a "$S=$R/bioscan512_a/$e/features_gpu/$e" \
        --b "$S=$R/bioscan512_b/$e/features_gpu/$e" --out "$R/checks/repeat_$e.json"
done
A=/flash/project_462001534/feature_geometry/benchmark/extract/fgvc_aircraft_trainval.parquet
run python scripts/audit_feature_null_space.py --cache $R/aircraft/dinov3_l/features_gpu/dinov3_l \
    --weights $H/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51/model.safetensors \
    --key-prefix norm --out $R/checks/rank_dinov3_l_aircraft.json
run python scripts/audit_feature_null_space.py --cache $R/aircraft/dinov2_l/features_gpu/dinov2_l \
    --weights $H/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/model.safetensors \
    --key-prefix layernorm --out $R/checks/rank_dinov2_l_aircraft.json
for e in bioclip2 siglip2; do
    run python scripts/audit_feature_null_space.py --cache $R/aircraft/$e/features_gpu/$e --out $R/checks/rank_${e}_aircraft.json
done
runs=""; for d in aircraft bioscan512_a bioscan512_b; do for e in dinov3_l dinov2_l bioclip2 siglip2; do runs="$runs ${d}_$e=$R/$d/$e"; done; done
run python scripts/benchmark/summarise_extraction.py --run $runs --out $R/checks/extraction_summary.csv
echo "end $(date -Is)"
