#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --qos=high_p


# Directory definitions
MAXNUMAT=500
#LMDB_DIR="/oak/stanford/groups/rondror/projects/atom3d/lmdb/RES/splits/split-by-class"
LMDB_DIR='/scratch/users/aderry/atom3d/res_bal_lmdb/all_tmp_0'
#NPZ_DIR="/oak/stanford/groups/rondror/projects/atom3d/npz/res/res_split-by-class_maxnumat-$MAXNUMAT"
NPZ_DIR="/oak/stanford/groups/rondror/projects/atom3d/npz/res/all_tmp_0"
# Create output directory
mkdir -p $NPZ_DIR
# Convert the dataset from LMDB to NPZ (using pre-defined split datasets)
python ../../atom3d/datasets/res/prepare_npz.py $LMDB_DIR $NPZ_DIR $MAXNUMAT --droph #--split


