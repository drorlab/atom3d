"""Code for preparing a pairs dataset (filtering and splitting)."""
import numpy as np
import pandas as pd
import click

import atom3d.datasets.ppi.neighbors as nb
import atom3d.filters.pdb
import atom3d.filters.sequence
import atom3d.protein.scop as scop
import atom3d.protein.sequence
import atom3d.protein.sequence as seq
import atom3d.splits.sequence
import atom3d.filters.filters as filters
import atom3d.shard.shard as sh
import atom3d.shard.shard_ops as sho
import atom3d.util.file as fi
import atom3d.util.log as log

logger = log.get_logger('prepare')


def split(input_sharded, output_root, shuffle_buffer, cutoff = 30):
    """Split by sequence identity."""
    if input_sharded.get_keys() != ['ensemble']:
        raise RuntimeError('Can only apply to sharded by ensemble.')

    all_chain_sequences = []
    logger.info('Loading chain sequences')
    for _, shard in input_sharded.iter_shards():
        all_chain_sequences.extend(seq.get_all_chain_sequences_df(shard))

    logger.info('Splitting by cluster')
    train, val, test = atom3d.splits.sequence.cluster_split(all_chain_sequences, cutoff)

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



@click.command(help='Prepare a sequence identity split.')
@click.argument('input_sharded_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--shuffle_buffer', type=int, default=10,
              help='How many shards to use in streaming shuffle. 0 means will '
              'not shuffle.')
@click.option('--cutoff', type=float, default=30,
              help='Cutoff (in %) for sequence identity.')
def prepare_seqid_split(input_sharded_path, output_root, shuffle_buffer, cutoff):
    input_sharded = sh.Sharded.load(input_sharded_path)
    split(input_sharded, output_root, shuffle_buffer, cutoff=cutoff)


if __name__ == "__main__":
    prepare_seqid_split()

