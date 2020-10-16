#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import pandas as pd

from atom3d.util import file as fi


def get_label(pdb, label_df):
    return label_df[label_df['pdb'] == pdb]['label'].iloc[0]

def main(datapath, out_path):
    valid_pdbs = [fi.get_pdb_code(f) for f in fi.find_files(out_path, 'sdf')]
    dat = []
    with open(os.path.join(datapath, 'index/INDEX_refined_data.2019')) as f:
        for line in f:
            if line.startswith('#'):
                continue
            l = line.strip().split()
            if l[0] not in valid_pdbs:
                continue
            dat.append(l[:5]+l[6:])
    refined_set = pd.DataFrame(dat, columns=['pdb','res','year','neglog_aff','affinity','ref','ligand'])

    refined_set[['measurement', 'affinity']] = refined_set['affinity'].str.split('=',expand=True)

    refined_set['ligand'] = refined_set['ligand'].str.strip('()')

    # Remove peptide ligands
    # - refined set size now 4,598

#     refined_set = refined_set[["-mer" not in l for l in refined_set.ligand]]
    
    
    refined_set.to_csv(os.path.join(out_path, 'pdbbind_refined_set_cleaned.csv'), index=False)

    labels = refined_set[['pdb', 'neglog_aff']].rename(columns={'neglog_aff': 'label'})

    labels.to_csv(os.path.join(out_path, 'pdbbind_refined_set_labels.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=str, help='directory where PDBBind is located')
    parser.add_argument('out_dir', type=str, help='directory to write label files')
    args = parser.parse_args()
    main(args.datapath, args.out_dir)



