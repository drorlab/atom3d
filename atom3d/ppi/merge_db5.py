"""Merge DB5 4-file format into single PDB structure with different models."""
import collections as col
import os

import Bio.PDB
import click

import atom3d.util.datatypes as dt
import atom3d.util.file as fi
import atom3d.util.log as log

logger = log.getLogger('merge')


@click.command(help='Merge 4-file DB5 format into single PDB format.')
@click.argument('path_to_db5', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def merge_db5(path_to_db5, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pdb_files = fi.find_files(path_to_db5, 'pdb')
    complexes = col.defaultdict(list)
    for f in pdb_files:
        pdb_code = os.path.basename(f)[:4]
        complexes[pdb_code].append(f)

    for pdb_code, files in complexes.items():
        if len(files) != 4:
            raise RuntimeError('Expect 4 files per pdb code in DB5.')
        receptor_bound = [x for x in files if '_r_b_' in os.path.basename(x)]
        receptor_unbound = [x for x in files if '_r_u_' in os.path.basename(x)]
        ligand_bound = [x for x in files if '_l_b_' in os.path.basename(x)]
        ligand_unbound = [x for x in files if '_l_u_' in os.path.basename(x)]
        if len(receptor_bound) != 1 or len(receptor_unbound) != 1 or \
                len(ligand_bound) != 1 or len(ligand_unbound) != 1:
            raise RuntimeError(
                f'{pdb_code:} does not have correct files present')
        logger.info(f'{pdb_code:}')

        receptor_bound = dt.read_pdb(receptor_bound[0])
        receptor_unbound = dt.read_pdb(receptor_unbound[0])
        ligand_bound = dt.read_pdb(ligand_bound[0])
        ligand_unbound = dt.read_pdb(ligand_unbound[0])
        if len(receptor_bound) != 1 or len(receptor_unbound) != 1 or \
                len(ligand_bound) != 1 or len(ligand_unbound) != 1:
            raise RuntimeError(
                f'{pdb_code:} has files with more than one model present')

        rb, ru, lb, lu = receptor_bound[0], receptor_unbound[0], \
            ligand_bound[0], ligand_unbound[0]
        lb.id, lb.serial_num = 0, 0
        lu.id, lu.serial_num = 1, 1
        rb.id, rb.serial_num = 2, 2
        ru.id, ru.serial_num = 3, 3
        output_name = f'{output_path:}/{pdb_code:}.pdb'
        bp = Bio.PDB.Structure.Structure(pdb_code)
        bp.add(lb)
        bp.add(lu)
        bp.add(rb)
        bp.add(ru)
        dt.write_pdb(output_name, bp)


if __name__ == "__main__":
    merge_db5()
