#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import numpy as np
import os
import subprocess
import sys
sys.path.append('..')
from util import datatypes as dt
from util import file as fi
from tqdm import tqdm
import argparse
from rdkit.Chem import PandasTools


def convert_to_hdf5(input_dir, label_file, hdf_file):
    cif_files = fi.find_files(input_dir, 'cif')
    proteins = []
    pockets = []
    pdb_codes = []
    for f in tqdm(cif_files, desc='reading structures'):
        pdb_code = fi.get_pdb_code(f)
        if '_protein' in f:
            pdb_codes.append(pdb_code)
            df = dt.bp_to_df(dt.read_any(f))
            proteins.append(df)
        elif '_pocket' in f:
            df = dt.bp_to_df(dt.read_any(f))
            pockets.append(df)
    
    print('converting proteins...')
    protein_df = pd.concat(proteins)
    pocket_df = pd.concat(pockets)
    pdb_codes = pd.DataFrame({'pdb': pdb_codes})
    
    protein_df.to_hdf(hdf_file, 'proteins')
    pocket_df.to_hdf(hdf_file, 'pockets')
    pdb_codes.to_hdf(hdf_file, 'pdb_codes')
    
    print('converting ligands...')
    sdf_files = fi.find_files(input_dir, 'sdf')
    big_sdf = os.path.join(input_dir, 'all_ligands.sdf')
    combine_sdfs(sdf_files, big_sdf)
    lig_df = PandasTools.LoadSDF(big_sdf, molColName='Mol')
    lig_df.index = pdb_codes
    lig_df.to_hdf(hdf_file, 'ligands')
    
    print('converting labels...')
    label_df = pd.read_csv(label_file)
    label_df = label_df.set_index('pdb').reindex(pdb_codes)
    label_df.to_hdf(hdf_file, 'labels')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=str, help='directory where data is located')
    parser.add_argument('label_file', type=str, help='path to label csv')
    parser.add_argument('out_file', type=str, default=6.0, help='output hdf5 file')
    args = parser.parse_args()
    convert_to_hdf5(args.datapath, args.label_file, args.out_file)

