#!/bin/bash

PDBBIND_DIR=$1
SPLITS_DIR=$2

# create PDBBind time splits
mkdir -p $SPLITS_DIR/time_split
python splitting.py time $PDBBIND_DIR --write_to_file

# create PDBBind core splits
mkdir -p $SPLITS_DIR/core_split
python splitting.py core $PDBBIND_DIR --write_to_file --cv_method random
python splitting.py core $PDBBIND_DIR --write_to_file --cv_method cluster --identity 30
python splitting.py core $PDBBIND_DIR --write_to_file --cv_method cluster --identity 90
