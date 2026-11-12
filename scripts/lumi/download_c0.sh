#!/bin/bash
# C0 downloads (task.md d57a630), CPU only:
#  - DF24 300-px images (DF20-300px.tar.gz; the README says DF24 uses the DF20 images) and the official DF24 metadata
#    zip, from the dataset authors' server; the zip must be byte-identical to the one the frozen DF24 split was built
#    from (sha256 8c56f095...); gzip test, sha256, and sha1 of every image member (content table);
#  - SigLIP 2 so400m-patch16-256 model.safetensors + config + processor at the pinned revision into the shared
#    hf_cache; the weights must match the Hugging Face LFS sha256 (810ddc85...).
# Usage: sbatch download_c0.sh CODE_DIR OUT_DF24
#SBATCH --job-name=fg_c0_dl
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
CODE=$1 OUT=$2
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST code $(cat "$CODE/CODE_COMMIT")"
source "$CODE/scripts/lumi/container_env.sh"
mkdir -p "$OUT"
B=http://ptak.felk.cvut.cz/plants/DanishFungiDataset
for f in DF20-300px.tar.gz DanishFungi2024.zip; do
    [ -s "$OUT/$f" ] || { curl -fSL --retry 5 --retry-delay 30 -o "$OUT/$f.part" "$B/$f" && mv "$OUT/$f.part" "$OUT/$f"; }
done
echo "8c56f095dfd07a19b93c23dd31c136998a34c03799ea7aecb870a7c7dd41baec  $OUT/DanishFungi2024.zip" | sha256sum -c -
gzip -t "$OUT/DF20-300px.tar.gz" && echo "gzip ok"
echo -e "DF20-300px.tar.gz\t$(stat -c %s "$OUT/DF20-300px.tar.gz")\t$(sha256sum "$OUT/DF20-300px.tar.gz" | cut -d' ' -f1)" | tee "$OUT/archives.tsv"
singularity exec "$SIF" python "$CODE/scripts/lumi/hash_archive_members.py" "$OUT/DF20-300px.tar.gz" --content "$OUT/content.tsv"
export HF_HOME=/scratch/project_462001534/aqua_v1/scratch_$USER/hf_cache
singularity exec "$SIF" python - <<'PY'
import hashlib
from huggingface_hub import hf_hub_download
repo, rev = "google/siglip2-so400m-patch16-256", "e8708ab72d125807e45b36fb7d4e0aacbb59f379"
for f in ("config.json", "preprocessor_config.json", "model.safetensors"):
    p = hf_hub_download(repo, f, revision=rev)
    print(f, p)
h = hashlib.sha256()
with open(p, "rb") as fh:
    for block in iter(lambda: fh.read(1 << 24), b""):
        h.update(block)
assert h.hexdigest() == "810ddc85ac019e6b2738b9130ae3602e41eb093a848a297a0ad296ca1f39dc67", h.hexdigest()
print("siglip2 model.safetensors sha256 matches the Hugging Face LFS record")
PY
echo "end $(date -Is)"
