"""Other operations for sharded datasets."""
import logging
import os
import random

import tqdm

import atom3d.shard.shard as sh
import atom3d.util.formats as dt


def filter_sharded(input_sharded, output_sharded, filter_fn, shuffle_buffer=0):
    """Filter sharded dataset to new sharded dataset, using provided filter."""
    logging.basicConfig(format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    if not os.path.exists(os.path.dirname(output_sharded.path)):
        os.makedirs(os.path.dirname(output_sharded.path))

    input_num_shards = input_sharded.get_num_shards()

    # We will just map to tmp, then reshard.
    tmp_path = output_sharded.get_prefix() + f'_tmp@{input_num_shards:}'
    tmp_sharded = sh.Sharded(tmp_path, input_sharded.get_keys())

    logging.info(f'Filtering {input_sharded.path:} to {output_sharded.path:}')
    # Apply filter.
    for shard_num in tqdm.trange(input_num_shards):
        df = input_sharded.read_shard(shard_num)
        if len(df) > 0:
            df = filter_fn(df)
        tmp_sharded._write_shard(shard_num, df)

    num_input_structures = input_sharded.get_num_keyed()
    num_output_structures = tmp_sharded.get_num_keyed()
    logging.info(f'After filtering, have {num_output_structures:} / '
                 f'{num_input_structures:} left.')
    reshard(tmp_sharded, output_sharded, shuffle_buffer)
    tmp_sharded.delete_files()


def reshard(input_sharded, output_sharded, shuffle_buffer=0):
    """
    Rebalance dataset, optionally shuffling.

    If shuffle_buffer is not 0, then we perform a streaming shuffle across
    shuffle_buffer number of output shards.
    """
    dirname = os.path.dirname(output_sharded.path)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(dirname, exist_ok=True)

    num_structures = input_sharded.get_num_keyed()
    output_num_shards = output_sharded.get_num_shards()
    input_num_shards = input_sharded.get_num_shards()

    shard_ranges = sh._get_shard_ranges(num_structures, output_num_shards)
    shard_sizes = shard_ranges[:, 1] - shard_ranges[:, 0]

    if shuffle_buffer != 0:
        buffer_size = shuffle_buffer * shard_sizes[0]
    else:
        buffer_size = 1

    t = tqdm.trange(output_num_shards)
    next_output_shard_num, next_input_shard_num = 0, 0
    to_write, to_consume = [], []
    df = None
    while True:
        while len(to_consume) < buffer_size and \
                (next_input_shard_num != input_num_shards):
            # Read next shard if need more examples.
            df = input_sharded.read_shard(next_input_shard_num)
            to_consume += [y for (_, y) in
                           dt.split_df(df, input_sharded.get_keys())]

            if shuffle_buffer is not None:
                random.shuffle(to_consume)
            next_input_shard_num += 1

        if len(to_consume) != 0:
            to_write.append(to_consume.pop(0))

        if len(to_write) == shard_sizes[next_output_shard_num]:
            # Write output shard if have number needed.

            if len(to_write) == 0:
                # Insert empty dataframe if nothing to write.
                to_write = [df.iloc[0:0]]

            output_sharded._write_shard(next_output_shard_num,
                                        dt.merge_dfs(to_write))
            to_write = []
            next_output_shard_num += 1
            t.update(1)

            if next_output_shard_num == output_num_shards:
                break


def rekey(input_sharded, output_sharded, shuffle_buffer=0):
    """Rekey dataset."""
    dirname = os.path.dirname(output_sharded.path)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(dirname, exist_ok=True)

    input_num_shards = input_sharded.get_num_shards()

    # We will just map to tmp, then reshard.
    tmp_path = output_sharded.get_prefix() + f'_tmp@{input_num_shards:}'
    tmp_sharded = sh.Sharded(tmp_path, output_sharded.get_keys())

    for shard_num in tqdm.trange(input_num_shards):
        df = input_sharded.read_shard(shard_num)
        tmp_sharded._write_shard(shard_num, df)

    num_input_structures = input_sharded.get_num_keyed()
    num_output_structures = tmp_sharded.get_num_keyed()
    logging.info(f'After rekey-ing, have {num_output_structures:} keyed, '
                 f'from {num_input_structures:} originally.')
    reshard(tmp_sharded, output_sharded, shuffle_buffer)
    tmp_sharded.delete_files()
