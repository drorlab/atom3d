#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --qos=high_p


# Directory definitions
LMDB_DIR=/oak/stanford/groups/rondror/projects/atom3d/lmdb/SMP/splits/random/data
NPZ_DIR=/oak/stanford/groups/rondror/projects/atom3d/npz/smp

# Create output directory
mkdir -p $NPZ_DIR
# Convert the dataset from LMDB to NPZ (using pre-defined split datasets)
python ../../atom3d/datasets/smp/prepare_npz.py $LMDB_DIR $NPZ_DIR --split \

