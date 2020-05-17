"""Code for preparing a lap dataset (filtering and splitting)."""
import click
import pandas as pd

import atom3d.util.log as log
import atom3d.util.filters as filters
import atom3d.util.shard as sh
import atom3d.util.shard_ops as sho
import atom3d.util.splits as splits


logger = log.getLogger('lap_prepare')


def split(input_sharded, output_root, info_csv):
    """Split randomly, balancing inactives and actives across sets."""
    info = pd.read_csv(info_csv)
    info['ensemble'] = info.apply(
        lambda x: x['ligand'] + '__' + x['active_struc'].split('_')[2] + '__' +
        x['inactive_struc'].split('_')[2], axis=1)
    info = info.set_index('ensemble')
    # Remove duplicate ensembles.
    info = info[~info.index.duplicated()]

    ensembles = input_sharded.get_names()
    in_use = info.loc[ensembles]
    active = in_use[in_use['label'] == 'A']
    inactive = in_use[in_use['label'] == 'I']

    a_test, a_val, a_train = splits.random_split(len(active))
    i_test, i_val, i_train = splits.random_split(len(inactive))

    train = active.iloc[a_train].index.tolist() + \
        inactive.iloc[i_train].index.tolist()
    val = active.iloc[a_val].index.tolist() + \
        inactive.iloc[i_val].index.tolist()
    test = active.iloc[a_test].index.tolist() + \
        inactive.iloc[i_test].index.tolist()

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

    sho.filter_sharded(input_sharded, train_sharded, train_filter_fn)
    sho.filter_sharded(input_sharded, val_sharded, val_filter_fn)
    sho.filter_sharded(input_sharded, test_sharded, test_filter_fn)


@click.command(help='Prepare lap dataset')
@click.argument('input_sharded_path', type=click.Path())
@click.argument('output_sharded_path', type=click.Path())
@click.argument('info_csv', type=click.Path(exists=True))
def prepare(input_sharded_path, output_sharded_path, info_csv):
    input_sharded = sh.load_sharded(input_sharded_path)
    output_sharded = sh.Sharded(output_sharded_path, input_sharded.get_keys())
    split(input_sharded, output_sharded, info_csv)


if __name__ == "__main__":
    prepare()
