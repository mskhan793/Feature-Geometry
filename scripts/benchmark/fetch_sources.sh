#!/bin/bash
# Download the original papers / official documentation pages of the benchmark datasets and of SigLIP 2 (read-only,
# small files) into OUT/<name>/, and write OUT/sha256.txt. Large data archives are NOT fetched here.
# Usage: fetch_sources.sh OUT_DIR
set -euo pipefail
OUT=$1
get() { mkdir -p "$OUT/$1"; curl -fsSL --retry 3 -A "Mozilla/5.0 (research data provenance)" -o "$OUT/$1/$2" "$3"; echo "$1/$2 <- $3" >> "$OUT/sources.txt"; }
mkdir -p "$OUT"; : > "$OUT/sources.txt"
get fgvc_aircraft paper_arxiv_1306.5151.pdf https://arxiv.org/pdf/1306.5151
get fgvc_aircraft official_page.html https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
get food101 paper_bossard_eccv14.pdf https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf
get food101 official_page.html https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
get cifar100 tech_report_krizhevsky2009.pdf https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
get cifar100 official_page.html https://www.cs.toronto.edu/~kriz/cifar.html
get oxford_pets paper_parkhi12a.pdf https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf
get oxford_pets official_page.html https://www.robots.ox.ac.uk/~vgg/data/pets/
get siglip2 paper_arxiv_2502.14786.pdf https://arxiv.org/pdf/2502.14786
# DF24 licence: the dataset authors' README at the commit read on 2026-11-13 (copy in evidence/benchmark_readiness/df24_licence)
get df24 README_at_c5c19b41.md https://raw.githubusercontent.com/BohemianVRA/DanishFungiDataset/c5c19b41275286d1ea8ba888224a2b68eb2132f7/README.md
(cd "$OUT" && find . -type f ! -name sha256.txt ! -name sources.txt | sort | xargs sha256sum > sha256.txt)
cat "$OUT/sha256.txt"
