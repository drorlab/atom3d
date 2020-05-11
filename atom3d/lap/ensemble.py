"""Code to ensemble ligand activity prediction (LAP) dataset."""
import collections as col
import os

import pandas as pd


def lap_ensembler(pdb_files):
    dirs = list(set([os.path.dirname(f) for f in pdb_files]))
    info_files = [os.path.join(x, 'info.csv') for x in dirs]
    labels = pd.concat([pd.read_csv(x) for x in info_files])

    pdbs = {os.path.splitext(os.path.basename(f))[0]: f for f in pdb_files}
    ligands = col.defaultdict(list)

    ensembles = {}
    for i, entry in labels.iterrows():
        lig = entry['ligand']
        active = entry['active_struc']
        inactive = entry['inactive_struc']
        name = lig + '__' + active.split('_')[2] + '__' + \
            inactive.split('_')[2]
        ensembles[name] = {
            active.split('_')[2] + '_active': pdbs[active],
            inactive.split('_')[2] + '_inactive': pdbs[inactive],
        }

    return ensembles
