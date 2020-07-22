"""Code for preparing a lep dataset (filtering and splitting)."""
import click
import pandas as pd

import atom3d.shard.filters as filters
import atom3d.shard.shard as sh
import atom3d.shard.shard_ops as sho
import atom3d.util.log as log
import atom3d.util.splits as splits

logger = log.get_logger('lep_prepare')


def split(input_sharded, output_root, info_csv, shuffle_buffer):
    """Split by protein."""
    if input_sharded.get_keys() != ['ensemble']:
        raise RuntimeError('Can only apply to sharded by ensemble.')

    info = pd.read_csv(info_csv)
    info['ensemble'] = info.apply(
        lambda x: x['ligand'] + '__' + x['active_struc'].split('_')[2] + '__' +
        x['inactive_struc'].split('_')[2], axis=1)
    info = info.set_index('ensemble')
    # Remove duplicate ensembles.
    info = info[~info.index.duplicated()]

    ensembles = input_sharded.get_names()['ensemble']
    in_use = info.loc[ensembles]
    active = in_use[in_use['label'] == 'A']
    inactive = in_use[in_use['label'] == 'I']

    # Split by protein.
    proteins = info['protein'].unique()
    i_test, i_val, i_train = splits.random_split(len(proteins), 0.6, 0.2, 0.2)
    p_train = proteins[i_train]
    p_val = proteins[i_val]
    p_test = proteins[i_test]
    logger.info(f'Train proteins: {p_train:}')
    logger.info(f'Val proteins: {p_val:}')
    logger.info(f'Test proteins: {p_test:}')

    train = info[info['protein'].isin(p_train)].index.tolist()
    val = info[info['protein'].isin(p_val)].index.tolist()
    test = info[info['protein'].isin(p_test)].index.tolist()

    logger.info(f'{len(train):} train examples, {len(val):} val examples, '
                f'{len(test):} test examples.')

    keys = input_sharded.get_keys()
    prefix = sh.get_prefix(output_root)
    num_shards = sh.get_num_shards(output_root)
    train_sharded = sh.Sharded(f'{prefix:}_train@{num_shards:}', keys)
    val_sharded = sh.Sharded(f'{prefix:}_val@{num_shards:}', keys)
    test_sharded = sh.Sharded(f'{prefix:}_test@{num_shards:}', keys)

    train_filter_fn = filters.form_filter_against_list(train, 'ensemble')
    val_filter_fn = filters.form_filter_against_list(val, 'ensemble')
    test_filter_fn = filters.form_filter_against_list(test, 'ensemble')

    sho.filter_sharded(
        input_sharded, train_sharded, train_filter_fn, shuffle_buffer)
    sho.filter_sharded(
        input_sharded, val_sharded, val_filter_fn, shuffle_buffer)
    sho.filter_sharded(
        input_sharded, test_sharded, test_filter_fn, shuffle_buffer)


@click.command(help='Prepare lep dataset')
@click.argument('input_sharded_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.argument('info_csv', type=click.Path(exists=True))
@click.option('--shuffle_buffer', type=int, default=5,
              help='How many shards to use in streaming shuffle. 0 means will '
              'not shuffle.')
def prepare(input_sharded_path, output_root, info_csv, shuffle_buffer):
    input_sharded = sh.Sharded.load(input_sharded_path)
    split(input_sharded, output_root, info_csv, shuffle_buffer)


if __name__ == "__main__":
    prepare()
