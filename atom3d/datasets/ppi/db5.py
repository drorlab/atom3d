"""DB5-specific functions."""
import collections as col
import os


def db5_ensembler(pdb_files):
    complexes = col.defaultdict(list)
    for f in pdb_files:
        pdb_code = os.path.basename(f)[:4]
        complexes[pdb_code].append(f)

    ensembles = {}
    for pdb_code, files in complexes.items():
        if len(files) != 4:
            raise RuntimeError('Expect 4 files per pdb code in DB5.')
        receptor_bound = [x for x in files if '_r_b' in os.path.basename(x)]
        receptor_unbound = [x for x in files if '_r_u' in os.path.basename(x)]
        ligand_bound = [x for x in files if '_l_b' in os.path.basename(x)]
        ligand_unbound = [x for x in files if '_l_u' in os.path.basename(x)]
        if len(receptor_bound) != 1 or len(receptor_unbound) != 1 or \
                len(ligand_bound) != 1 or len(ligand_unbound) != 1:
            raise RuntimeError(
                f'{pdb_code:} does not have correct files present')

        ensembles[pdb_code] = {
            pdb_code + '_receptor_bound': receptor_bound[0],
            pdb_code + '_receptor_unbound': receptor_unbound[0],
            pdb_code + '_ligand_bound': ligand_bound[0],
            pdb_code + '_ligand_unbound': ligand_unbound[0],
        }

    return ensembles
