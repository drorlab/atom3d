"""Generate protein interfaces labels for DB5 datasets."""
import collections as col
import logging
import os

import click

import atom3d.ppi.gen_labels as gl
import atom3d.util.datatypes as dt
import atom3d.util.file as fi


@click.command(help='Find neighbors for DB5 dataset.')
@click.argument('path_to_db5', type=click.Path(exists=True))
@click.option('-c', '--cutoff', type=int, default=8,
              help='Maximum distance (in angstroms), for two residues to be '
              'considered neighbors.')
@click.option('--cutoff-type', default='CA',
              type=click.Choice(['heavy', 'CA'], case_sensitive=False),
              help='How to compute distance between residues: CA is based on '
              'alpha-carbons, heavy is based on any heavy atom.')
def gen_labels_db5(path_to_db5, cutoff, cutoff_type):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

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
        logging.info(f'{pdb_code:}')
        receptor_bound = dt.bp_to_df(dt.read_pdb(receptor_bound[0]))
        receptor_unbound = dt.bp_to_df(dt.read_pdb(receptor_unbound[0]))
        ligand_bound = dt.bp_to_df(dt.read_pdb(ligand_bound[0]))
        ligand_unbound = dt.bp_to_df(dt.read_pdb(ligand_unbound[0]))
        logging.info('loaded')
        neighbors = gl.get_all_neighbors(
            [ligand_unbound, receptor_unbound],
            [ligand_bound, receptor_bound],
            cutoff, cutoff_type)


if __name__ == "__main__":
    gen_labels_db5()
