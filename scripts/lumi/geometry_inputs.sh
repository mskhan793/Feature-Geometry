# Inputs of the full-specimen geometry jobs on LUMI (task.md 92a605a), sourced by geometry_folds.sh and
# geometry_summary.sh with CODE and DATASET set. Sets the container ($SIF, bindings), the selection ($COHORT), the
# verified feature caches ($CACHES, MANIFEST=CACHE pairs), the source column ($SOURCE) and component-metadata
# arguments ($META_ARGS), and checks that every input is readable inside the container.
source "$CODE/scripts/lumi/container_env.sh"
F=/flash/project_462001534/feature_geometry
S=/scratch/project_462001534/feature_geometry
case $DATASET in
    finbenthic2)
        COHORT=$F/finbenthic2/full_selection.parquet
        CACHES="$F/finbenthic2/cohort_hdf5.parquet=$S/finbenthic2_fold0/23623689/features_gpu/dinov3_l \
$F/finbenthic2/full_new_images.parquet=$S/full_specimen/finbenthic2/extract/features_gpu/dinov3_l"
        SOURCE=none
        META_ARGS="--fields --numeric log_image_area" ;;
    aquamonitor)
        COHORT=$F/aquamonitor/full_selection.parquet
        CACHES="$COHORT=$S/full_specimen/aquamonitor/extract/features_gpu/dinov3_l"
        SOURCE=dataset
        META_ARGS="" ;;
    *) echo "unknown dataset $DATASET" >&2; exit 1 ;;
esac
for f in "$CODE/scripts/class_geometry_pilot.py" "$COHORT" $(echo "$CACHES" | tr ' ' '\n' | tr '=' '\n'); do
    singularity exec "$SIF" test -r "$f" || { echo "not readable in the container: $f" >&2; exit 1; }
done
