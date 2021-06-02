#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --qos=high_p


# Directory definitions
MAXNUMAT=500
LMDB_DIR="/oak/stanford/groups/rondror/projects/atom3d/lmdb/LEP/splits/split-by-protein"
for CUTOFF in 40 45 50 55 60; do
	NPZ_DIR="/oak/stanford/groups/rondror/projects/atom3d/npz/lep/lep_split-by-protein_cutoff-${CUTOFF}_maxnumat-$MAXNUMAT"
	# Create output directory
	mkdir -p $NPZ_DIR
	# Convert the dataset from LMDB to NPZ (using pre-defined split datasets)
	python ../../atom3d/datasets/lep/prepare_npz.py $LMDB_DIR $NPZ_DIR $MAXNUMAT $CUTOFF --split --droph
done

