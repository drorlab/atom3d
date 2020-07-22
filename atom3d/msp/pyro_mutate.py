import csv
import os
import sys

import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.toolbox import mutate_residue


def read_csv(csv_file):
    rows = []
    muts_from_csv = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
    return rows

def pyro_mutate(rows, inputpdbs, outputdir):
    counter = 0
    for i in range(len(rows)):
        if '1KBH' in rows[i][0] or '1JCK' in rows[i][0]: #skip 1KBH, 1JCK
            print('skipping {}'.format(rows[i])) 
            continue
        pdb, mutstr = rows[i][0], rows[i][1]
        fromres = mutstr[0]
        chain = mutstr[1]
        tores = mutstr[-1]
        resnum = mutstr[2:-1]

        new_file = pdb + '_' + mutstr + '.pdb' 

        if os.path.exists(os.path.join(outputdir, new_file)):
            print('{} was already handled'.format(rows[i]))
            counter += 1
            continue
        
        p = pose_from_pdb(os.path.join(inputpdbs, pdb + '.pdb'))
        pyro_resi = p.pdb_info().pdb2pose(chain, int(resnum))

        mutate_residue(p, pyro_resi, tores, pack_radius=10.0)
        p.dump_pdb(os.path.join(outputdir, new_file))
        counter += 1
        print('Done with {} counter: {}'.format(new_file, i))

    print('Done mutating all files')
    print(str(counter) + ' files mutated.')

def main():
    pyrosetta.init()
    inputcsv = sys.argv[1]
    inputpdbs = sys.argv[2]
    outputdir = sys.argv[3]
    rows = read_csv(inputcsv)
    pyro_mutate(rows, inputpdbs, outputdir)

main()