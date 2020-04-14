"""Generate protein interfaces labels for DIPS dataset."""
import logging
import os

import click

import atom3d.ppi.gen_labels as gl
import atom3d.util.datatypes as dt
import atom3d.util.file as fi


@click.command(help='Find neighbors for DIPS dataset.')
@click.argument('path_to_dips', type=click.Path(exists=True))
@click.option('-c', '--cutoff', type=int, default=8,
              help='Maximum distance (in angstroms), for two residues to be '
              'considered neighbors.')
@click.option('--cutoff-type', default='CA',
              type=click.Choice(['heavy', 'CA'], case_sensitive=False),
              help='How to compute distance between residues: CA is based on '
              'alpha-carbons, heavy is based on any heavy atom.')
def gen_labels_dips(path_to_dips, cutoff, cutoff_type):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    pdb_files = fi.find_files(path_to_dips, 'pdb*.gz')
    print(len(pdb_files))
    for f in pdb_files:
        logging.info(f'{os.path.basename(f):}')
        df = dt.bp_to_df(dt.read_pdb(f))
        neighbors = gl.get_all_neighbors(
            [df], [], cutoff, cutoff_type)


if __name__ == "__main__":
    gen_labels_dips()
