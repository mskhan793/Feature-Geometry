#!/bin/bash
# Download the official archives of FGVC-Aircraft (2013b), Food-101 and Oxford-IIIT Pet (images + annotations) to
# OUT/<dataset>/ on LUMI, verify the publisher's MD5 where one is published (FGVC-Aircraft), record sha256 of every
# archive, and list the archive members (tar -tzf) so manifests can check image availability without unpacking.
# CIFAR-100 is copied from the Dell separately (its MD5 is verified there and here). Archives are not extracted.
#
# Usage: sbatch download_benchmark_sources.sh OUT
#SBATCH --job-name=fg_bench_dl
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
OUT=$1
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST"
fetch() {  # dataset file url
    mkdir -p "$OUT/$1"
    [ -s "$OUT/$1/$2" ] || curl -fSL --retry 5 --retry-delay 30 -o "$OUT/$1/$2.part" "$3"
    [ -s "$OUT/$1/$2" ] || mv "$OUT/$1/$2.part" "$OUT/$1/$2"
    echo "$1/$2 <- $3" >> "$OUT/sources.txt"
}
fetch fgvc_aircraft fgvc-aircraft-2013b.tar.gz https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
fetch food101 food-101.tar.gz http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
fetch oxford_pets images.tar.gz https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
fetch oxford_pets annotations.tar.gz https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
echo "32eca553f897e747706144f3bb444df0  $OUT/fgvc_aircraft/fgvc-aircraft-2013b.tar.gz" | md5sum -c -
[ -s "$OUT/cifar100/cifar-100-python.tar.gz" ] && echo "eb9058c3a382ffc7106e4002c42a8d85  $OUT/cifar100/cifar-100-python.tar.gz" | md5sum -c -
for f in "$OUT"/*/*.tar.gz; do
    tar -tzf "$f" > "$f.members.txt"
    echo "$(sha256sum "$f" | cut -d' ' -f1)  $(stat -c %s "$f")  ${f#$OUT/}  $(wc -l < "$f.members.txt") members"
done | tee "$OUT/archives.tsv"
echo "end $(date -Is)"
