"""Generate protein interfaces labels for sharded dataset."""
import warnings

import click
import pandas as pd
import parallel as par

import atom3d.ppi.neighbors as nb
import atom3d.util.log as log
import atom3d.util.shard as sh

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

logger = log.getLogger('genLabels')


@click.command(help='Find neighbors for sharded dataset.')
@click.argument('sharded_path', type=click.Path())
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
def get_neighbors_sharded(sharded_path, cutoff, cutoff_type, num_threads,
                          overwrite):
    sharded = sh.Sharded.load(sharded_path)
    num_shards = sh.get_num_shards(sharded_path)

    requested_shards = list(range(num_shards))
    if not overwrite:
        produced_shards = [x for x in requested_shards
                           if sharded.has(x, 'neighbors')]
    else:
        produced_shards = []

    work_shards = set(requested_shards).difference(produced_shards)
    logger.info(f'{len(requested_shards):} requested, '
                f'{len(produced_shards):} already produced, '
                f'{len(work_shards):} left to do.')
    logger.info(f'Using {num_threads:} threads')

    inputs = [(sharded, shard_num, cutoff, cutoff_type)
              for shard_num in work_shards]

    par.submit_jobs(_gen_labels_shard, inputs, num_threads)


def _gen_labels_shard(sharded, shard_num, cutoff, cutoff_type):
    logger.info(f'Processing shard {shard_num:}')
    shard = sharded.read_shard(shard_num)

    all_neighbors = []
    for e, ensemble in shard.groupby('ensemble'):
        neighbors = nb.neighbors_from_ensemble(ensemble, cutoff, cutoff_type)
        all_neighbors.append(neighbors)
    all_neighbors = pd.concat(all_neighbors).reset_index(drop=True)
    sharded.add_to_shard(shard_num, all_neighbors, 'neighbors')
    logger.info(f'Done processing shard {shard_num:}')


if __name__ == "__main__":
    get_neighbors_sharded()
