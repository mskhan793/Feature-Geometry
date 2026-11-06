# Sourced by every LUMI job script that runs Python in the LAIF container. Sets $SIF and loads the bindings module:
# without lumi-aif-singularity-bindings the container cannot see /scratch, /flash or /projappl ("No such file").
module --force purge
module load Local-LAIF lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260731_122833/lumi-multitorch-full-u24r70f21m50t210-20260731_122833.sif
export SIF
