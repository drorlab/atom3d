"""Code for preparing a pairs dataset (splitting)."""
import click

import atom3d.util.filters as filters
import atom3d.util.log as log
import atom3d.util.sequence as seq
import atom3d.util.shard as sh
import atom3d.util.shard_ops as sho
import atom3d.util.splits as splits

logger = log.getLogger('prepare')


def split(input_sharded, output_root, shuffle_buffer):
    """Split by sequence identity."""
    if input_sharded.get_keys() != ['ensemble']:
        raise RuntimeError('Can only apply to sharded by ensemble.')

    all_chain_sequences = []
    logger.info('Loading chain sequences')
    for _, shard in input_sharded.iter_shards():
        all_chain_sequences.extend(seq.get_all_chain_sequences_df(shard))

    logger.info('Splitting by cluster')
    train, val, test = splits.cluster_split(all_chain_sequences, 30)

    # Will just look up ensembles.
    train = [x[0] for x in train]
    val = [x[0] for x in val]
    test = [x[0] for x in test]

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


@click.command(help='Split pair dataset')
@click.argument('input_sharded_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('-b', '--bsa', default=None,
              help='File to use for bsa filtering.')
@click.option('--against_path', default=None,
              help='Sharded dataset to filter against (for SCOP and seq)')
@click.option('--shuffle_buffer', type=int, default=10,
              help='How many shards to use in streaming shuffle. 0 means will '
              'not shuffle.')
def filter_pairs(input_sharded_path, output_root, bsa, against_path,
                 shuffle_buffer):
    input_sharded = sh.load_sharded(input_sharded_path)
    keys = input_sharded.get_keys()
    if keys != ['ensemble']:
        raise RuntimeError('Can only apply to sharded by ensemble.')
    split(input_sharded, output_root, shuffle_buffer)


if __name__ == "__main__":
    filter_pairs()
