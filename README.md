# atom3d
Atomic tasks on molecules in three dimensions


Google Document:
https://docs.google.com/document/d/1JGk5nxvH7AZ4YdL81KqvIVZ7_xWlpM4G4BEsz6ikwVE

## Installation

On sherlock, first create a conda environment:

`conda create -n atom3d python=3.6 pip`

To install required packages:

```
conda activate atom3d
make requirements
```

## Sample usage

To split a PDB for the protein interface prediction (ppi) dataset:

```
ROOT_DIR=/oak/stanford/groups/rondror/projects/atom3d/protein_interface_prediction/DB5
python -m atom3d.ppi.gen_labels $ROOT_DIR/1A2K/1A2K_*_u_cleaned.pdb -b $ROOT_DIR/1A2K/1A2K_l_b_cleaned.pdb -b $ROOT_DIR/1A2K/1A2K_r_b_cleaned.pdb test.labels
```
