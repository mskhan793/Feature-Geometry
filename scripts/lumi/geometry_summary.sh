#!/bin/bash
# After the five fold runs of both datasets (task.md 92a605a), on one LUMI CPU node:
#   1. output A: descriptive_geometry.py on fold0 train + val pooled (in-sample, exploratory; fold0 test not read);
#   2. code regression: the earlier capped FIN-Benthic2 fold-0 cohort rerun with this code, compared with the original
#      run (ROOT/finbenthic2/baseline_original, copied from the Dell);
#   3. real-image montages of the fold0 selection (train specimens only);
#   4. AquaMonitor LUMI features vs the earlier Dell pilot features on their 196,983 shared images
#      (ROOT/aquamonitor/dell_caches, copied from the Dell);
#   5. summarise_five_folds.py: per-fold, across-fold and cross-dataset tables, with the earlier capped fold-0 runs
#      (ROOT/<dataset>/baseline_original) as baselines.
#
# Usage: sbatch geometry_summary.sh CODE_DIR ROOT
#SBATCH --job-name=fg_geom_summary
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail  # no -u: the module scripts read unset variables

export CODE=$1 ROOT=$2
N=$SLURM_CPUS_PER_TASK
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST code $(cat "$CODE/CODE_COMMIT")"
declare -A H5=([finbenthic2]=/flash/project_462001534/feature_geometry/finbenthic2/finbenthic2_images.h5
               [aquamonitor]=/flash/project_462001534/data_hdf_aqua/aquamonitor_complete.h5)
declare -A NUMERIC=([finbenthic2]="--numeric log_image_area" [aquamonitor]="")
for DATASET in finbenthic2 aquamonitor; do
    export DATASET
    source "$CODE/scripts/lumi/geometry_inputs.sh"
    ENV="export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$N; cd $CODE"
    D=$ROOT/$DATASET
    singularity exec "$SIF" bash -c "$ENV; python scripts/descriptive_geometry.py --cohort $COHORT --caches $CACHES \
        --out $D/descriptive_fold0 --fold fold0 --source-column $SOURCE ${NUMERIC[$DATASET]} --jobs $N \
        > $D/descriptive_fold0.log 2>&1"
    singularity exec "$SIF" bash -c "$ENV; python scripts/class_geometry_montage.py --cohort $COHORT \
        --hdf5 ${H5[$DATASET]} --selection $D/geometry/fold0/selection.json --out $D/geometry/fold0 --fold fold0 \
        > $D/montage_fold0.log 2>&1"
    echo "$DATASET done $(date -Is)"
done
F=/flash/project_462001534/feature_geometry/finbenthic2
B=$ROOT/finbenthic2
singularity exec "$SIF" bash -c "$ENV; python scripts/class_geometry_pilot.py --cohort $F/cohort_hdf5.parquet \
    --caches $F/cohort_hdf5.parquet=/scratch/project_462001534/feature_geometry/finbenthic2_fold0/23623689/features_gpu/dinov3_l \
    --out $B/baseline_rerun --source-column none --jobs $N > $B/baseline_rerun.log 2>&1; \
    python scripts/compare_datasets_geometry.py --a $B/baseline_original --a-name original_6c4c515 \
    --b $B/baseline_rerun --b-name rerun --out $B/baseline_regression > $B/baseline_regression.log 2>&1"
A=$ROOT/aquamonitor/dell_caches  # the earlier AquaMonitor pilot features (Dell RTX 4090), copied for this check only
singularity exec "$SIF" bash -c "$ENV; python scripts/compare_caches_by_name.py --names $A/expanded_cohort.parquet \
    --a /flash/project_462001534/feature_geometry/aquamonitor/full_selection.parquet=/scratch/project_462001534/feature_geometry/full_specimen/aquamonitor/extract/features_gpu/dinov3_l \
    --b $A/camera_pilot_cohort.parquet=$A/camera_pilot/dinov3_l $A/expanded_new_images.parquet=$A/expanded/dinov3_l \
    --out $ROOT/aquamonitor/lumi_vs_dell.json > $ROOT/aquamonitor/lumi_vs_dell.log 2>&1"
singularity exec "$SIF" bash -c "$ENV; python scripts/summarise_five_folds.py \
    --dataset FIN-Benthic2 $ROOT/finbenthic2/geometry $ROOT/finbenthic2/baseline_original log_image_area \
    --dataset AquaMonitor $ROOT/aquamonitor/geometry $ROOT/aquamonitor/baseline_original dataset \
    --out $ROOT/summary > $ROOT/summary.log 2>&1"
echo "end $(date -Is)"
