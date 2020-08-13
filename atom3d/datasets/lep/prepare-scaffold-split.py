"""Code for preparing a scaffold-split dataset."""
import numpy as np
import pandas as pd
import click

import atom3d.filters.pdb
import atom3d.filters.sequence
import atom3d.splits.splits as splits
import atom3d.filters.filters as filters
import atom3d.shard.shard as sh
import atom3d.shard.shard_ops as sho
import atom3d.util.file as fi
import atom3d.util.log as log

logger = log.get_logger('prepare')


def split(input_sharded, output_root, scaffold_data, shuffle_buffer):
    """Split by scaffold."""
    if input_sharded.get_keys() != ['ensemble']:
        raise RuntimeError('Can only apply to sharded by ensemble.')

    logger.info('Process scaffold and ensemble data')
    scaffold_list = scaffold_data['Scaffold'].tolist()
    ensemble_list = []
    for i in range(len(scaffold_data)):
        ensemble = ''
        ensemble += scaffold_data.iloc[i]['ligand']
        ensemble += '__'+scaffold_data.iloc[i]['active_struc'].split('_')[-1]
        ensemble += '__'+scaffold_data.iloc[i]['inactive_struc'].split('_')[-1]
        ensemble_list.append(ensemble)
    ensemble_list = np.array(ensemble_list)
 
    logger.info('Splitting by scaffold')
    train_idx, val_idx, test_idx = splits.scaffold_split(scaffold_list)
    train = ensemble_list[train_idx].tolist()
    val = ensemble_list[val_idx].tolist()
    test = ensemble_list[test_idx].tolist()

    keys = input_sharded.get_keys()
    if keys != ['ensemble']:
        raise RuntimeError('Can only apply to sharded by ensemble.')
    prefix = sh.get_prefix(output_root)
    num_shards = sh.get_num_shards(output_root)
    train_sharded = sh.Sharded(f'{prefix:}_train@{num_shards:}', keys)
    val_sharded = sh.Sharded(f'{prefix:}_val@{num_shards:}', keys)
    test_sharded = sh.Sharded(f'{prefix:}_test@{num_shards:}', keys)

    logger.info('Writing sets')
    train_filter_fn = filters.form_filter_against_list(train, 'ensemble')
    val_filter_fn = filters.form_filter_against_list(val, 'ensemble')
    test_filter_fn = filters.form_filter_against_list(test, 'ensemble')

    sho.filter_sharded(
        input_sharded, train_sharded, train_filter_fn, shuffle_buffer)
    sho.filter_sharded(
        input_sharded, val_sharded, val_filter_fn, shuffle_buffer)
    sho.filter_sharded(
        input_sharded, test_sharded, test_filter_fn, shuffle_buffer)

    # write splits to text files
    np.savetxt(output_root.split('@')[0]+'_train.txt', train, fmt='%s')
    np.savetxt(output_root.split('@')[0]+'_val.txt', val, fmt='%s')
    np.savetxt(output_root.split('@')[0]+'_test.txt', test, fmt='%s')


@click.command(help='Prepare a sequence identity split.')
@click.argument('input_sharded_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.argument('scaffold_file', type=click.Path(exists=True))
@click.option('--shuffle_buffer', type=int, default=10,
              help='How many shards to use in streaming shuffle. 0 means will '
              'not shuffle.')
def prepare_scaffold_split(input_sharded_path, output_root, shuffle_buffer, scaffold_file):
    input_sharded = sh.Sharded.load(input_sharded_path)
    scaffold_data = pd.read_csv(scaffold_file)
    split(input_sharded, output_root, scaffold_data, shuffle_buffer)


if __name__ == "__main__":
    prepare_scaffold_split()

