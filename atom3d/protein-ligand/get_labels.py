#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import sys
sys.path.append('..')
from util import datatypes as dt
import argparse

def main(datapath):
    dat = []
    with open(os.path.join(datapath, 'refined-set/index/INDEX_refined_data.2019')) as f:
        for line in f:
            if line.startswith('#'):
                continue
            l = line.strip().split()
            dat.append(l[:5]+l[6:])
    refined_set = pd.DataFrame(dat, columns=['pdb','res','year','neglog_aff','affinity','ref','ligand'])

    refined_set[['measurement', 'affinity']] = refined_set['affinity'].str.split('=',expand=True)

    refined_set['ligand'] = refined_set['ligand'].str.strip('()')

    # Remove peptide ligands
    # - refined set size now 4,598

    refined_set = refined_set[["-mer" not in l for l in refined_set.ligand]]

    refined_set.to_csv(os.path.join(datapath, 'pdbbind_refined_set_cleaned.csv'), index=False)

    labels = refined_set[['pdb', 'ligand', 'neglog_aff']].rename(columns={'neglog_aff': 'label'})

    labels.to_csv(os.path.join(datapath, 'pdbbind_refined_set_labels.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=str, help='directory where PDBBind is located')
    args = parser.parse_args()
    main(args.datapath)



