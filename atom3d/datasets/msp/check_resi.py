import csv
import glob
import os
import sys

import pyrosetta
from pyrosetta import pose_from_pdb

pyrosetta.init()
files = sorted(glob.glob('../pyrosetta_mut/*.pdb'))
match = 0
nomatchfiles = []

jobid = int(sys.argv[1])
num_jobs = int(sys.argv[2])
perjob = int(len(files) / int(num_jobs))

nums = list(range(len(files)))
if jobid == num_jobs - 1:
    indices = nums[(jobid)*perjob:]
else:
    indices = nums[jobid*perjob: (jobid+1)*perjob]
print(indices)

log = []
for idx in indices:
    s = os.path.basename(files[idx])
    temp = s.split('_')
    fromres = temp[-1][0]
    chain = temp[-1][1]
    tores = temp[-1][-5]
    resnum = temp[-1][2:-5]
    p = pose_from_pdb(files[idx])
    pyro_resi = p.pdb_info().pdb2pose(chain, int(resnum))
    pyro_resname = p.residue(pyro_resi).name1()
    if pyro_resname == tores:
        match += 1
    else:
        nomatchfiles.append(files[idx])
    log.append([temp, fromres, chain, resnum, tores, pyro_resname])

print(match, len(indices), nomatchfiles, len(nomatchfiles))
with open('checkresiloginprogram{}.txt'.format(jobid), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(log)