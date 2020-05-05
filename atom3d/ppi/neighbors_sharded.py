"""Generate protein interfaces labels for sharded dataset."""
import warnings

import click
import pandas as pd
import parallel as par

import atom3d.ppi.neighbors as nb
import atom3d.util.shard as sh
import atom3d.util.log as log

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

logger = log.getLogger('genLabels')


@click.command(help='Find neighbors for sharded dataset.')
@click.argument('sharded', type=click.Path())
@click.option('-c', '--cutoff', type=int, default=8,
              help='Maximum distance (in angstroms), for two residues to be '
              'considered neighbors.')
@click.option('--cutoff-type', default='CA',
              type=click.Choice(['heavy', 'CA'], case_sensitive=False),
              help='How to compute distance between residues: CA is based on '
              'alpha-carbons, heavy is based on any heavy atom.')
@click.option('-n', '--num_threads', default=8,
              help='Number of threads to use for parallel processing.')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Overwrite existing neighbors.')
@click.option('--db5/--no-db5', default=False,
              help='Whether files are in DB5 merged format.')
def get_neighbors_sharded(sharded, cutoff, cutoff_type, num_threads, overwrite,
                          db5):
    num_shards = sh.get_num_shards(sharded)

    requested_shards = list(range(num_shards))
    if not overwrite:
        produced_shards = [x for x in requested_shards
                           if sh.has(sharded, x, 'neighbors')]
    else:
        produced_shards = []

    work_shards = set(requested_shards).difference(produced_shards)
    logger.info(f'{len(requested_shards):} requested, '
                f'{len(produced_shards):} already produced, '
                f'{len(work_shards):} left to do.')
    logger.info(f'Using {num_threads:} threads')

    inputs = [(sharded, shard_num, cutoff, cutoff_type, db5)
              for shard_num in work_shards]

    par.submit_jobs(_gen_labels_shard, inputs, num_threads)


def _gen_labels_shard(sharded, shard_num, cutoff, cutoff_type, db5):
    logger.info(f'Processing shard {shard_num:}')
    shard = sh.read_shard(sharded, shard_num)

    all_neighbors = []
    for structure, x in shard.groupby('structure'):
        if db5:
            lb = x[x['model'] == 0].copy()
            lu = x[x['model'] == 1].copy()
            rb = x[x['model'] == 2].copy()
            ru = x[x['model'] == 3].copy()
            lb['model'] = 1
            rb['model'] = 3
            neighbors = nb.get_neighbors(
                [lu, ru], [lb, rb], cutoff, cutoff_type)
        else:
            neighbors = nb.get_neighbors(
                [x], [], cutoff, cutoff_type)
        all_neighbors.append(neighbors)
    all_neighbors = pd.concat(all_neighbors).reset_index(drop=True)
    sh.add_to_shard(sharded, shard_num, all_neighbors, 'neighbors')
    logger.info(f'Done processing shard {shard_num:}')


if __name__ == "__main__":
    get_neighbors_sharded()
