"""Label mutation pairs as beneficial or detrimental."""

import click
import pandas as pd
import parallel as par

import atom3d.util.log as log
import atom3d.util.shard as sh

logger = log.getLogger('msp_label')


@click.command(help='Label SKEMPI pairs with good/bad label.')
@click.argument('sharded_path', type=click.Path())
@click.argument('data_csv', type=click.Path(exists=True))
@click.option('-n', '--num_threads', default=8,
              help='Number of threads to use for parallel processing.')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Overwrite existing labels.')
def gen_labels_sharded(sharded_path, data_csv, num_threads, overwrite):
    sharded = sh.load_sharded(sharded_path)
    num_shards = sharded.get_num_shards()

    requested_shards = list(range(num_shards))
    if not overwrite:
        produced_shards = [x for x in requested_shards
                           if sharded.has(x, 'labels')]
    else:
        produced_shards = []

    work_shards = set(requested_shards).difference(produced_shards)
    logger.info(f'{len(requested_shards):} requested, '
                f'{len(produced_shards):} already produced, '
                f'{len(work_shards):} left to do.')
    logger.info(f'Using {num_threads:} threads')

    inputs = [(sharded, shard_num, data_csv)
              for shard_num in work_shards]

    par.submit_jobs(_gen_labels_shard, inputs, num_threads)


def _gen_labels_shard(sharded, shard_num, data_csv):
    logger.info(f'Processing shard {shard_num:}')
    shard = sharded.read_shard(shard_num)
    data = pd.read_csv(data_csv, names=['oname', 'mutation', 'label'])
    data['ensemble'] = data.apply(
        lambda x: x['oname'] + '_' + x['mutation'], axis=1)
    data['chain'] = data['mutation'].apply(lambda x: x[1])
    data['residue'] = data['mutation'].apply(lambda x: int(x[2:-1]))
    data['original_resname'] = data['mutation'].apply(lambda x: x[0])
    data['mutated_resname'] = data['mutation'].apply(lambda x: x[-1])
    data = data.set_index('ensemble', drop=False)

    all_labels = []
    for e, ensemble in shard.groupby('ensemble'):
        all_labels.append(data.loc[e][['label', 'ensemble', 'chain', 'residue',
                                       'original_resname', 'mutated_resname']])
    all_labels = pd.concat(all_labels, axis=1).T.reset_index(drop=True)
    sharded.add_to_shard(shard_num, all_labels, 'labels')
    logger.info(f'Done processing shard {shard_num:}')


if __name__ == "__main__":
    gen_labels_sharded()
