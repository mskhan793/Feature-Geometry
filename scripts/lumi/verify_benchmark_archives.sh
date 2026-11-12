#!/bin/bash
# Verify the downloaded benchmark archives (download_benchmark_sources.sh) and prepare manifest inputs:
#  - gzip integrity of every archive; md5 + sha256 recorded (archives.tsv);
#  - FGVC-Aircraft: a second download of the full archive (hashes compared with the first) and the separately published
#    annotations-only archive with its MD5 (split files compared with those inside the full archive);
#  - per archive: sha1 of every image member (content tables) and extraction of the small split/annotation files only.
# Usage: sbatch verify_benchmark_archives.sh OUT CODE_DIR
#SBATCH --job-name=fg_bench_verify
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
OUT=$1 CODE=$2
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST code $(cat "$CODE/CODE_COMMIT")"
source "$CODE/scripts/lumi/container_env.sh"
A=$OUT/fgvc_aircraft
curl -fsSL --retry 5 -o "$A/second_download.tar.gz" https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
curl -fsSL --retry 5 -o "$A/fgvc-aircraft-2013b-annotations.tar.gz" https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b-annotations.tar.gz
curl -fsSL --retry 5 -o "$A/published_md5_full.txt" https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.md5
curl -fsSL --retry 5 -o "$A/published_md5_annotations.txt" https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b-annotations.tar.gz.md5 || true
for f in "$OUT"/*/*.tar.gz; do
    gzip -t "$f" && ok=ok || ok=CORRUPT
    echo -e "${f#$OUT/}\t$(stat -c %s "$f")\t$(md5sum "$f" | cut -d' ' -f1)\t$(sha256sum "$f" | cut -d' ' -f1)\tgzip_$ok"
done | tee "$OUT/archives.tsv"
mkdir -p "$A/annotations_only"
tar -xzf "$A/fgvc-aircraft-2013b-annotations.tar.gz" -C "$A/annotations_only"
run() { singularity exec "$SIF" python "$CODE/scripts/lumi/hash_archive_members.py" "$@"; }
run "$A/fgvc-aircraft-2013b.tar.gz" --content "$A/content.tsv" --extract '/data/[^/]+\.txt$' --extract-to "$A/extracted"
run "$OUT/food101/food-101.tar.gz" --content "$OUT/food101/content.tsv" --extract '/meta/' --extract-to "$OUT/food101/extracted"
run "$OUT/oxford_pets/images.tar.gz" --content "$OUT/oxford_pets/content.tsv"
mkdir -p "$OUT/oxford_pets/extracted" && tar -xzf "$OUT/oxford_pets/annotations.tar.gz" -C "$OUT/oxford_pets/extracted" annotations/trainval.txt annotations/test.txt annotations/list.txt
diff -r <(cd "$A/extracted"/*/data && sha256sum *.txt) <(cd "$A/annotations_only"/*/data && sha256sum *.txt) \
    && echo "aircraft split files: full archive == annotations-only archive" || echo "aircraft split files DIFFER"
echo "end $(date -Is)"
