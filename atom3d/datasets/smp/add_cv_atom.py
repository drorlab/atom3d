import numpy as np
import pandas as pd

import atom3d.util.formats as dt

in_dir_name = '../../data/qm9/raw'
csv_file = in_dir_name+'/gdb9.sdf.csv'
sdf_file = in_dir_name+'/gdb9.sdf'
out_file = in_dir_name+'/gdb9_with_cv_atom.csv'

df = pd.read_csv(csv_file)
raw_data = [ df[col] for col in df.keys() ]
raw_mols = dt.read_sdf_to_mol(sdf_file, sanitize=False)

raw_num_atoms = []
for im, m in enumerate(raw_mols):
    if m is None:
        print('Molecule',im+1,'could not be processed.')
        new_numat = None
        continue
    new_numat = m.GetNumAtoms()
    raw_num_atoms.append(new_numat)
raw_num_atoms = np.array(raw_num_atoms)

df['cv_atom'] = df['cv'] - (raw_num_atoms*2.981)

print('cv:')
print(' mean:', np.mean(df['cv']))
print(' sdev:', np.std(df['cv']))
print('cv_atom:')
print(' mean:', np.mean(df['cv_atom']))
print(' sdev:', np.std(df['cv_atom']))

df.to_csv(out_file, index=False)

