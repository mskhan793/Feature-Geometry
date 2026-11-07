#!/bin/bash
# One smoke-test task (one GCD, one encoder), started by smoke_baselines.sh through srun. Steps, each logged with its
# exit status and duration: v1 equivalence gate (512 images), extraction of the first $ROWS manifest rows in chunks of
# $CHUNK, cache verification, rerun (must skip every chunk), repeat of chunk 0 into a second cache and comparison.
# The smoke rows are a software check only, not a research partition. Exit status is non-zero if any step failed.
set -o pipefail
ENCODERS=(bioclip2 dinov2_l dinov3_l)
E=${ENCODERS[$SLURM_PROCID]}
OUT=$SMOKE/$E
mkdir -p "$OUT"
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
RUN="singularity exec $SIF bash -c"
ENV="export PYTHONPATH=$CODE:$V1/pkgs HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 ROCR_VISIBLE_DEVICES=$SLURM_LOCALID; cd $CODE"
status=0
step() {  # name, command
    local t0=$(date +%s)
    $RUN "$ENV; $2" > "$OUT/$1.log" 2>&1
    local rc=$?
    echo "{\"encoder\": \"$E\", \"step\": \"$1\", \"exit\": $rc, \"seconds\": $(( $(date +%s) - t0 ))}" | tee -a "$OUT/steps.jsonl"
    [ $rc -eq 0 ] || status=1
}
declare -A REF=([bioclip2]=$V1/runs/m3_s42_baseline/feature_cache [dinov2_l]=$V1/frozen_backbones/dinov2_vitl14_frozen/cache
                [dinov3_l]=$V1/frozen_backbones/dinov3_vitl16_exact_frozen/cache)
X="python scripts/extract_features.py --manifest $MANIFEST --hdf5 $HDF5 --encoder $E --chunk-size $CHUNK --task 0 --n-tasks 1 --num-workers 6"

if [ "$SLURM_PROCID" = 0 ]; then
    step encoder_support "python scripts/check_encoder_support.py --out $SMOKE/encoder_support_lumi.json"
fi
step v1_gate "python scripts/compare_with_v1_cache.py --encoder $E --v1-parquet $V1_PARQUET --v1-cache ${REF[$E]} --split val --hdf5 $HDF5 --n 512 --out $OUT/v1_gate.json"
step extract "$X --cache-dir $OUT/cache_a --limit-rows $ROWS"
step verify "python scripts/verify_feature_cache.py --manifest $MANIFEST --cache-dir $OUT/cache_a/$E --encoder $E --chunk-size $CHUNK --limit-rows $ROWS --out $OUT/verify.json"
step resume "$X --cache-dir $OUT/cache_a --limit-rows $ROWS"
skipped=$(grep -c "already done" "$OUT/resume.log")
expected=$(( (ROWS + CHUNK - 1) / CHUNK ))
echo "{\"encoder\": \"$E\", \"step\": \"resume_skipped\", \"chunks_skipped\": $skipped, \"expected\": $expected}" | tee -a "$OUT/steps.jsonl"
[ "$skipped" -eq "$expected" ] || status=1
step repeat "$X --cache-dir $OUT/cache_b --limit-rows $CHUNK"
step repeat_compare "python scripts/compare_feature_caches.py --a $OUT/cache_a/$E --b $OUT/cache_b/$E --out $OUT/repeat.json"
exit $status
