"""Generate BSA for DIPS datasets."""
import warnings

import click
import pandas as pd
import parallel as par

import atom3d.ppi.bsa as bsa
import atom3d.util.log as log
import atom3d.util.shard as sh

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


logger = log.getLogger('bsa')


@click.command(help='Get Buried Surface Area for DIPS dataset.')
@click.argument('sharded', type=click.Path())
@click.option('-n', '--num_threads', default=8,
              help='Number of threads to use for parallel processing.')
@click.option('-o', '--overwrite/--no-overwrite', default=False,
              help='Overwrite existing neighbors.')
@click.option('--db5/--no-db5', default=False,
              help='Whether files are in DB5 merged format.')
def bsa_dips(sharded, num_threads, overwrite, db5):
    num_shards = sh.get_num_shards(sharded)

    requested_shards = list(range(num_shards))
    if not overwrite:
        produced_shards = [x for x in requested_shards
                           if sh.has(sharded, x, 'bsa')]
    else:
        produced_shards = []

    work_shards = set(requested_shards).difference(produced_shards)
    inputs = [(sharded, x, db5) for x in work_shards]
    logger.info(f'{len(requested_shards):} requested, '
                f'{len(produced_shards):} already produced, '
                f'{len(work_shards):} left to do.')
    logger.info(f'Using {num_threads:} threads')

    par.submit_jobs(_bsa_shard, inputs, num_threads)


def _bsa_shard(sharded, shard_num, db5):
    logger.info(f'Processing shard {shard_num:}')
    shard = sh.read_shard(sharded, shard_num)

    all_results = []
    for structure, x in shard.groupby('structure'):
        if db5:
            lb = x[x['model'] == 0].copy()
            rb = x[x['model'] == 2].copy()
            lb['model'] = 1
            rb['model'] = 3
            result = bsa.compute_all_bsa([lb, rb], [lb, rb])
        else:
            result = bsa.compute_all_bsa([x], [])
        all_results.append(result)
    all_results = pd.concat(all_results).reset_index(drop=True)
    sh.add_to_shard(sharded, shard_num, all_results, 'bsa')
    logger.info(f'Done processing shard {shard_num:}')


if __name__ == "__main__":
    bsa_dips()
