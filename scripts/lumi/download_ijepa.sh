#!/bin/bash
# I-JEPA ViT-H/14 checkpoint gate (task.md 3b897b0), CPU only: download the official checkpoint
# (dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar, 10,358,004,345 bytes), check it against the server's
# multipart ETag, and store the target encoder as safetensors with a provenance record
# (scripts/benchmark/ijepa_target_weights.py). Output: OUT/IN1K-vit.h.14-300e.pth.tar (kept for provenance),
# OUT/IN1K-vit.h.14-300e_target_encoder.safetensors, OUT/target_weights_record.json.
# Usage: sbatch download_ijepa.sh CODE_DIR OUT
#SBATCH --job-name=fg_ijepa_dl
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
CODE=$1 OUT=$2
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST code $(cat "$CODE/CODE_COMMIT")"
source "$CODE/scripts/lumi/container_env.sh"
mkdir -p "$OUT"
URL=https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
CK=$OUT/IN1K-vit.h.14-300e.pth.tar
[ -s "$CK" ] || { curl -fSL --retry 5 --retry-delay 30 -C - -o "$CK.part" "$URL" && mv "$CK.part" "$CK"; }
[ "$(stat -c %s "$CK")" = 10358004345 ] || { echo "unexpected size $(stat -c %s "$CK")" >&2; exit 1; }
singularity exec "$SIF" bash -c "export PYTHONPATH=$CODE; cd $CODE; python scripts/benchmark/ijepa_target_weights.py \
    --checkpoint $CK --etag b0e94c3423adcbda5a2108a6c1544263-1235 \
    --out $OUT/IN1K-vit.h.14-300e_target_encoder.safetensors --record $OUT/target_weights_record.json"
echo "end $(date -Is)"
