#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --qos=high_p


# Directory definitions
MAXNUMAT=500
LMDB_DIR="/oak/stanford/groups/rondror/projects/atom3d/lmdb/ligand_binding_affinity/split-by-sequence-identity-30"
NPZ_DIR="/oak/stanford/groups/rondror/projects/atom3d/npz/lba/lba_split-by-sequence-identity-30_maxnumat-$MAXNUMAT"
# Create output directory
mkdir -p $NPZ_DIR
# Convert the dataset from LMDB to NPZ (using pre-defined split datasets)
python ../../atom3d/datasets/lba/prepare_npz.py $LMDB_DIR $NPZ_DIR $MAXNUMAT --split --droph


