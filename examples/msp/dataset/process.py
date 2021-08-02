import csv
import os
import sys
from collections import defaultdict

import pymol as py


#Functions that filter data
#==========================

#Get rid of duplicate pdb, mut
def filter_dup_lines(lines):
    new_lines = []
    done = set()
    for i in range(len(lines)):
        if tuple(lines[i][:3]) in done:
            continue
        new_lines.append(lines[i])
        done.add(tuple(lines[i][:3]))
    print('{} lines after getting rid of duplicates'.format(len(new_lines)))
    return new_lines

#We only want point mutations.
def filter_multiple_muts(lines):
    single = []
    notsingle = []
    for l in lines:
        if ',' in l[1] or ' ' in l[1].strip():
            notsingle.append(l)
        else:
            single.append(l)
    print('{} lines after getting rid of multiple mutations'.format(len(single)))
    return single

#Some mutants didn't form a complex at all.
def filter_nonbinders(lines):
    filtered = []
    for l in lines:
        if 'n.b.' not in l[6:10] and 'n.b' not in l[6:10] and '' not in l[6:10]:
            filtered.append(l)
    print('{} lines after getting rid of nonbinders'.format(len(filtered)))
    return filtered

#=========================

#1 if the mutation was affinity increasing and 0 otherwise.
def get_labels(lines):
    labels = []
    counts = defaultdict(int)
    for l in lines:
        mut_Kd = float(l[7])
        wt_Kd = float(l[9])
        if mut_Kd < wt_Kd: #These columns are Kd, not Ka, so smaller is a better binder!
            labels.append(1)
        else:
            labels.append(0)
        counts[l[3]] += 1
    print('{} mutations out of {} were increasing'.format(sum(labels), len(labels)))
    for loc in counts:
        print('{} in location: {}'.format(counts[loc], loc))
    print('')
    return labels

def shorten_lines(lines, label):
    lines_short = []
    for i, l in enumerate(lines):
        lines_short.append([l[0], l[2], label[i]])
    return lines_short

#Functions that use PyMol
def initiate():
    import __main__                       
    __main__.pymol_argv = ['pymol', '-qc']
    py.finish_launching() 
    py.cmd.set('group_auto_mode', 2)

def process_pdbs(lines, input_dir, write_dir):
    if not os.path.exists(input_dir):
        print('The path {} does not exist'.format(input_dir))
        return
    written = set()
    counter = 0
    for l in lines:
        py.cmd.reinitialize()

        pdb, chains1, chains2 = l[0].split('_')
        chains = chains1 + chains2
        sel = '('
        py.cmd.load(os.path.join(input_dir, pdb + '.pdb'))
        for let in ' '.join(chains).split():
            sel += 'chain ' + let + ' or '
        sel = sel[:-3]
        sel += ') and (r. '
        aa3 = 'ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split()
        sel += '+'.join(aa3)
        sel += ')'
        if tuple([pdb, chains1, chains2]) not in written:
            if not os.path.exists(os.path.join(write_dir, pdb + '_' + chains1 + '_' + chains2 + '.pdb')):
                save_path = os.path.join(write_dir, pdb + '_' + chains1 + '_' + chains2 + '.pdb')
                py.cmd.save(save_path, sel)
                print('Actually Wrote {} to {}'.format(pdb, write_dir))
            print('{} handled'.format(pdb))
        written.add(tuple([pdb, chains1, chains2]))
        counter += 1
    print('{} pdbs handled for {}'.format(counter, write_dir))


def terminate():
    py.cmd.quit()

def main():
    skempicsv = sys.argv[1]
    input_dir = sys.argv[2]
    write_dir = sys.argv[3]

    with open(skempicsv, 'r') as f:
        lines = [l.strip().split(';') for l in f]
    lines = lines[1:] #get rid of header

    print('{} total lines before filtering'.format(len(lines)))
    lines = filter_multiple_muts(lines)
    lines = filter_nonbinders(lines)
    lines = filter_dup_lines(lines)
    labels = get_labels(lines)

    lines_short = shorten_lines(lines, labels)

    #Clean PDBs
    initiate()
    process_pdbs(lines, input_dir, write_dir)
    terminate()

    #Write lines
    with open(os.path.join(write_dir, 'filtered_skempi.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(lines)

    #write lines short
    with open(os.path.join(write_dir, 'data.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(lines_short)

main()