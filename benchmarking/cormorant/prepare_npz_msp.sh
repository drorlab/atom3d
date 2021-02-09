#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --qos=high_p


# Directory definitions
for CUTOFF in 06 07 08 09 10 12 15 20 25; do
	LMDB_DIR="/oak/stanford/groups/rondror/projects/atom3d/lmdb/MSP/splits/split-by-seqid30/data"
	NPZ_DIR="/oak/stanford/groups/rondror/projects/atom3d/npz/msp/msp_split-by-sequence-identity-30_cutoff-${CUTOFF}"
	# Create output directory
	mkdir -p $NPZ_DIR
	# Convert the dataset from LMDB to NPZ (using pre-defined split datasets)
	python ../../atom3d/datasets/msp/prepare_npz.py $LMDB_DIR $NPZ_DIR $CUTOFF --split --droph
done

