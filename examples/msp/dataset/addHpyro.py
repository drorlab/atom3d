import os
from glob import glob

import pyrosetta
from pyrosetta import pose_from_pdb

pyrosetta.init()

pdbs = glob('../cleaned/*.pdb')

for f in pdbs:
    p = pose_from_pdb(f)
    # print(os.path.join('../cleaned_pyro', os.path.basename(f)))
    p.dump_pdb(os.path.join('../cleaned_pyro', os.path.basename(f)))
    print('Processed {}'.format(f))