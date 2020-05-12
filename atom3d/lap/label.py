"""Label pairs as active or inactive."""
import click
import pandas as pd
import parallel as par

import atom3d.util.log as log
import atom3d.util.shard as sh


logger = log.getLogger('lap_label')


@click.command(help='Label LAP pairs with inactive/active label.')
@click.argument('sharded', type=click.Path())
@click.argument('info_csv', type=click.Path(exists=True))
@click.option('-n', '--num_threads', default=8,
              help='Number of threads to use for parallel processing.')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Overwrite existing labels.')
def gen_labels_sharded(sharded, info_csv, num_threads, overwrite):
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

    inputs = [(sharded, shard_num, info_csv)
              for shard_num in work_shards]

    par.submit_jobs(_gen_labels_shard, inputs, num_threads)


def _gen_labels_shard(sharded, shard_num, info_csv):
    logger.info(f'Processing shard {shard_num:}')
    shard = sh.read_shard(sharded, shard_num)
    info = pd.read_csv(info_csv)
    info['ensemble'] = info.apply(
        lambda x: x['ligand'] + '__' + x['active_struc'].split('_')[2] + '__' +
        x['inactive_struc'].split('_')[2], axis=1)
    info = info.set_index('ensemble', drop=False)
    # Remove duplicate ensembles.
    info = info[~info.index.duplicated()]

    all_labels = []
    for e, ensemble in shard.groupby('ensemble'):
        all_labels.append(info.loc[e][['label', 'ensemble']])
    all_labels = pd.concat(all_labels, axis=1).T.reset_index(drop=True)
    sh.add_to_shard(sharded, shard_num, all_labels, 'labels')
    logger.info(f'Done processing shard {shard_num:}')


if __name__ == "__main__":
    gen_labels_sharded()
