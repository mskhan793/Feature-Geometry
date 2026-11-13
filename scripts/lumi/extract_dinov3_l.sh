#!/bin/bash
# Frozen DINOv3-L features for one manifest of HDF5 images on one LUMI node (FIN-Benthic2 fold-0 cohort, task.md
# 366ee62; full-specimen selections of FIN-Benthic2 and AquaMonitor, task.md 92a605a): the encoder-agnostic
# extract_encoder.sh with ENCODER=dinov3_l and --hdf5. Since C2 (task.md d57a630) the cache verification and the
# CPU-vs-GPU check are not part of the GPU job: run check_extraction.sh afterwards.
#
# Usage: sbatch [--partition P --time T] extract_dinov3_l.sh CODE_DIR MANIFEST HDF5 OUT_DIR
#SBATCH --job-name=fg_extract
#SBATCH --account=project_462001534
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=240G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
exec bash "$1/scripts/lumi/extract_encoder.sh" "$1" dinov3_l "$2" --hdf5 "$3" "$4"
