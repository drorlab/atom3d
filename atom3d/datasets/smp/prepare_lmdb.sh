#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rbaltman,owners
# # SBATCH --qos=high_p


# Directory definitions
OUT_DIR=/scratch/users/aderry/lmdb/atom3d/small_molecule_properties
XYZ_DIR=$SCRATCH/dsgdb9nsd
START_DIR=$(pwd)

cd $SCRATCH
# Check if the dataset exists already
if [ "$(ls dsgdb9nsd | wc -l)" -lt "133885" ]; then
	# Download and unpack the QM9 dataset
	wget -c https://ndownloader.figshare.com/files/3195389
	mv 3195389 dsgdb9nsd.xyz.tar.bz2
	mkdir -p $XYZ_DIR
	tar -xvjf dsgdb9nsd.xyz.tar.bz2 -C $XYZ_DIR
	# replace floating-point exponential 
	# (pandas can only read E notation)
	cd $XYZ_DIR
	for FILE in $(ls); do 
		sed -i 's/\*\^-/E-/g' $FILE
	done
fi
# Return to the start directory
cd $START_DIR
# Create output directory
mkdir -p $OUT_DIR
# Convert the dataset to LMDB using IDs from the "splits" folder
python prepare_lmdb.py $XYZ_DIR $OUT_DIR --split \
	--train_txt splits/ids_training.txt \
	--val_txt   splits/ids_validation.txt \
	--test_txt  splits/ids_test.txt 
# Remove the raw data  
# rm $SCRATCH/dsgdb9nsd.xyz.tar.bz2
# rm -r $XYZ_DIR

