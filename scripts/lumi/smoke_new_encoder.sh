#!/bin/bash
# Stage B smoke test (task.md 8538df5): one new encoder (DINOv2-L, cached weights) on one new dataset (CIFAR-100, a
# fixed sample of training images; scripts/benchmark/make_smoke_manifest.py). Two independent extractions into two
# caches are compared by image name (must be identical), and the final-LayerNorm rank constraint is audited with the
# checkpoint's own norm weights. Software / reproducibility check only; no validation or test data.
#
# Usage: sbatch smoke_new_encoder.sh CODE_DIR SMOKE_MANIFEST CIFAR_TAR OUT
#SBATCH --job-name=fg_bench_smoke
#SBATCH --account=project_462001534
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
CODE=$1 SMOKE=$2 CIFAR_TAR=$3 OUT=$4
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST code $(cat "$CODE/CODE_COMMIT")"
source "$CODE/scripts/lumi/container_env.sh"
export HF_HOME=/scratch/project_462001534/aqua_v1/scratch_muhamkha/hf_cache HF_HUB_OFFLINE=1
mkdir -p "$OUT"
tar -xzf "$CIFAR_TAR" -C "$OUT"
for f in "$CODE/scripts/extract_features.py" "$SMOKE" "$OUT/cifar-100-python/train"; do
    singularity exec "$SIF" test -r "$f" || { echo "not readable in the container: $f" >&2; exit 1; }
done
W=$(ls $HF_HOME/hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/model.safetensors)
run() { singularity exec "$SIF" bash -c "export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HOME=$HF_HOME HF_HUB_OFFLINE=1; cd $CODE; $*"; }
for rep in a b; do
    t0=$(date +%s)
    run python scripts/extract_features.py --manifest "$SMOKE" --cifar-dir "$OUT/cifar-100-python" --encoder dinov2_l \
        --cache-dir "$OUT/cache_$rep" --batch-size 128 --num-workers 6 > "$OUT/extract_$rep.log" 2>&1
    echo "extraction $rep: $(( $(date +%s) - t0 )) s"
done
run python scripts/compare_caches_by_name.py --names "$SMOKE" --a "$SMOKE=$OUT/cache_a/dinov2_l" --b "$SMOKE=$OUT/cache_b/dinov2_l" \
    --out "$OUT/repeat_comparison.json"
run python scripts/audit_feature_null_space.py --cache "$OUT/cache_a/dinov2_l" --weights "$W" --key-prefix layernorm \
    --out "$OUT/null_space_dinov2_l_cifar100.json"
rm -rf "$OUT/cifar-100-python"
echo "end $(date -Is)"
