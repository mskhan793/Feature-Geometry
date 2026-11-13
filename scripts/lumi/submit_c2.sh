#!/bin/bash
# C2 staged extraction (task.md d57a630), submitted from a LUMI login node: one extract_encoder.sh job (standard-g,
# 8 GCDs, time limit from the plan) per new encoder x dataset combination in PLAN (plan.tsv from
# scripts/benchmark/c2_plan.py), each followed by its check_extraction.sh CPU job (afterok). A rerun with the same
# plan resumes: finished chunks are skipped.
# Usage: submit_c2.sh CODE_DIR PLAN_TSV [DATASET ...]   (optional datasets restrict the batch)
set -euo pipefail
CODE=$1 PLAN=$2
shift 2
LOGS=/scratch/project_462001534/feature_geometry/benchmark_c2/logs
mkdir -p "$LOGS"
cd "$LOGS"
while IFS=$'\t' read -r enc ds limit manifest flag source out; do
    if [ $# -gt 0 ] && [[ ! " $* " =~ " $ds " ]]; then continue; fi
    id=$(sbatch --parsable --job-name="fg_c2_${ds}_$enc" --time="$limit" "$CODE/scripts/lumi/extract_encoder.sh" \
        "$CODE" "$enc" "$manifest" "$flag" "$source" "$out")
    chk=$(sbatch --parsable --job-name="fg_c2chk_${ds}_$enc" --dependency="afterok:$id" --kill-on-invalid-dep=yes \
        "$CODE/scripts/lumi/check_extraction.sh" "$CODE" "$enc" "$manifest" "$flag" "$source" "$out")
    printf '%s\t%s\t%s\t%s\t%s\n' "$enc" "$ds" "$id" "$chk" "$out" | tee -a "$LOGS/submitted.tsv"
done < "$PLAN"
