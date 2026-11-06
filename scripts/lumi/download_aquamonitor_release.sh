#!/bin/bash
# Download the AquaMonitor release (Hugging Face mikkoim/aquamonitor, pinned revision) into $DEST and verify every
# file against the manifest (sha256 for large files, git blob sha1 for small ones). Resumable: files listed in
# $DEST/verified.tsv are skipped; a file is moved into place only after its checksum matches. Exits non-zero unless
# every manifest file is verified.
#
# Usage: sbatch download_aquamonitor_release.sh MANIFEST.tsv DEST
#SBATCH --job-name=fg_aqua_download
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

MANIFEST=$(realpath "$1")
DEST=$2
REVISION=9a30834874d0428be529ab0d35d8aa3a687dd43a
BASE="https://huggingface.co/datasets/mikkoim/aquamonitor/resolve/$REVISION"
PARALLEL=4
mkdir -p "$DEST"
cd "$DEST"
touch verified.tsv

fetch() {  # path size algorithm checksum
    local path=$1 size=$2 algo=$3 want=$4 got
    if grep -qxF "$path" <(cut -f1 verified.tsv); then return 0; fi
    mkdir -p "$(dirname "$path")"
    curl -sSfL --retry 10 --retry-delay 30 --retry-all-errors -C - -o "$path.part" "$BASE/$path"
    if [ "$(stat -c %s "$path.part")" != "$size" ]; then echo "SIZE MISMATCH $path" >&2; return 1; fi
    if [ "$algo" = sha256 ]; then
        got=$(sha256sum "$path.part" | cut -d' ' -f1)
    else
        got=$( { printf 'blob %d\0' "$size"; cat "$path.part"; } | sha1sum | cut -d' ' -f1)
    fi
    if [ "$got" != "$want" ]; then echo "CHECKSUM MISMATCH $path" >&2; rm -f "$path.part"; return 1; fi
    mv "$path.part" "$path"
    printf '%s\t%s\t%s\n' "$path" "$algo" "$got" >> verified.tsv
}
export -f fetch
export BASE

echo "start $(date -Is) revision $REVISION dest $DEST"
tail -n +2 "$MANIFEST" | xargs -P "$PARALLEL" -L 1 bash -c 'fetch "$@"' _ || true

expected=$(tail -n +2 "$MANIFEST" | wc -l)
done_n=$(cut -f1 verified.tsv | sort -u | wc -l)
cp "$MANIFEST" manifest.tsv
echo "end $(date -Is): verified $done_n of $expected files, $(du -sb . | cut -f1) bytes"
[ "$done_n" -eq "$expected" ]
