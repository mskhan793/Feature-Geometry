#!/bin/bash
# After the five fold runs of both datasets (task.md 92a605a), on one LUMI CPU node:
#   1. output A: descriptive_geometry.py on fold0 train + val pooled (in-sample, exploratory; fold0 test not read);
#   2. real-image montages of the fold0 selection (train specimens only);
#   3. the earlier capped fold-0 pilots rerun with this code as baselines (ROOT/<dataset>/baseline): FIN-Benthic2 cohort
#      with its LUMI fold-0 cache, AquaMonitor expanded cohort with its Dell features (INPUTS/aquamonitor/dell_caches),
#      each compared with the original run (INPUTS/<dataset>/baseline_original);
#   4. AquaMonitor LUMI features vs the earlier Dell pilot features on their 196,983 shared images;
#   5. summarise_five_folds.py: per-fold, across-fold and cross-dataset tables, with the rerun baselines.
# INPUTS (copies from the Dell) = /scratch/project_462001534/feature_geometry/full_specimen; ROOT holds the fold runs
# (ROOT/<dataset>/geometry/foldK) and receives all outputs.
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
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST code $(cat "$CODE/CODE_COMMIT")"  # before any setup
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
INPUTS=/scratch/project_462001534/feature_geometry/full_specimen
A=$INPUTS/aquamonitor/dell_caches  # the earlier AquaMonitor pilot features (Dell RTX 4090)
F=/flash/project_462001534/feature_geometry/finbenthic2
declare -A BASE_ARGS=(
    [finbenthic2]="--cohort $F/cohort_hdf5.parquet --caches $F/cohort_hdf5.parquet=/scratch/project_462001534/feature_geometry/finbenthic2_fold0/23623689/features_gpu/dinov3_l --source-column none"
    [aquamonitor]="--cohort $A/expanded_cohort.parquet --caches $A/camera_pilot_cohort.parquet=$A/camera_pilot/dinov3_l $A/expanded_new_images.parquet=$A/expanded/dinov3_l")
declare -A BASE_COHORT=([finbenthic2]=$F/cohort_hdf5.parquet [aquamonitor]=$A/expanded_cohort.parquet)
declare -A META=([finbenthic2]="--fields --numeric log_image_area" [aquamonitor]="")
for DATASET in finbenthic2 aquamonitor; do
    B=$ROOT/$DATASET/baseline
    singularity exec "$SIF" bash -c "$ENV; python scripts/class_geometry_pilot.py ${BASE_ARGS[$DATASET]} --out $B \
        --jobs $N > $B.log 2>&1; python scripts/component_metadata.py --cohort ${BASE_COHORT[$DATASET]} \
        --components $B/specimen_components.csv --modality $B/per_class_modality.csv --geometry $B/per_class_geometry.csv \
        --out $B ${META[$DATASET]} >> $B.log 2>&1; python scripts/compare_datasets_geometry.py \
        --a $INPUTS/$DATASET/baseline_original --a-name original --b $B --b-name corrected \
        --out $ROOT/$DATASET/baseline_original_vs_corrected >> $B.log 2>&1"
done
singularity exec "$SIF" bash -c "$ENV; python scripts/compare_caches_by_name.py --names $A/expanded_cohort.parquet \
    --a /flash/project_462001534/feature_geometry/aquamonitor/full_selection.parquet=/scratch/project_462001534/feature_geometry/full_specimen/aquamonitor/extract/features_gpu/dinov3_l \
    --b $A/camera_pilot_cohort.parquet=$A/camera_pilot/dinov3_l $A/expanded_new_images.parquet=$A/expanded/dinov3_l \
    --out $ROOT/aquamonitor/lumi_vs_dell.json > $ROOT/aquamonitor/lumi_vs_dell.log 2>&1"
singularity exec "$SIF" bash -c "$ENV; python scripts/summarise_five_folds.py \
    --dataset FIN-Benthic2 $ROOT/finbenthic2/geometry $ROOT/finbenthic2/baseline log_image_area \
    --dataset AquaMonitor $ROOT/aquamonitor/geometry $ROOT/aquamonitor/baseline dataset \
    --out $ROOT/summary > $ROOT/summary.log 2>&1"
echo "end $(date -Is)"
