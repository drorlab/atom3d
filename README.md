# atom3d
Atomic tasks on molecules in three dimensions

## Installation

On sherlock, first create a conda environment:

`conda create -n atom3d python=3.6 pip`

To install required packages:

```
conda activate atom3d
make requirements
```

To setup environment configuration (e.g., to point to blast executables):

```
make env
```

Then fill in appropriate values .env

## Sample usage

To split a PDB for the protein interface prediction (ppi) dataset:

```
ROOT_DIR=/PATH/TO/protein_interface_prediction/DB5
python -m atom3d.ppi.gen_labels $ROOT_DIR/1A2K/1A2K_*_u_cleaned.pdb -b $ROOT_DIR/1A2K/1A2K_l_b_cleaned.pdb -b $ROOT_DIR/1A2K/1A2K_r_b_cleaned.pdb test.labels
```
