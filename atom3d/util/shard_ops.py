"""Other operations for sharded datasets."""
import logging
import os

import tqdm

import atom3d.util.datatypes as dt
import atom3d.util.shard as sh


def filter_sharded(input_sharded, output_sharded, filter_fn):
    """Filter sharded dataset to new sharded dataset, using provided filter."""
    logging.basicConfig(format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    if not os.path.exists(os.path.dirname(output_sharded)):
        os.makedirs(os.path.dirname(output_sharded))

    input_num_shards = sh.get_num_shards(input_sharded)

    # We will just map to tmp, then reshard.
    tmp_sharded = sh._get_prefix(output_sharded) + f'_tmp@{input_num_shards:}'

    logging.info(f'Filtering {input_sharded:} to {output_sharded:}')
    # Apply filter.
    for shard_num in tqdm.trange(input_num_shards):
        df = sh.read_shard(input_sharded, shard_num)
        df = filter_fn(df)
        sh._write_shard(tmp_sharded, shard_num, df)

    num_input_structures = sh.get_num_structures(input_sharded)
    num_output_structures = sh.get_num_structures(tmp_sharded)
    logging.info(f'After filtering, have {num_output_structures:} / '
                 f'{num_input_structures:} left.')
    reshard(tmp_sharded, output_sharded)
    sh.delete(tmp_sharded)


def reshard(input_sharded, output_sharded):
    """Reshard dataset."""
    dirname = os.path.dirname(output_sharded)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(dirname, exist_ok=True)

    num_structures = sh.get_num_structures(input_sharded)
    output_num_shards = sh.get_num_shards(output_sharded)
    input_num_shards = sh.get_num_shards(input_sharded)

    shard_ranges = sh._get_shard_ranges(num_structures, output_num_shards)
    shard_sizes = shard_ranges[:, 1] - shard_ranges[:, 0]

    t = tqdm.trange(output_num_shards)
    next_output_shard_num, next_input_shard_num = 0, 0
    to_write, to_consume = [], []
    while True:
        if len(to_consume) == 0 and (next_input_shard_num != input_num_shards):
            # Read next shard if need more examples.
            df = sh.read_shard(input_sharded, next_input_shard_num)
            to_consume = [y for (_, y) in dt.split_df(df)]
            next_input_shard_num += 1

        if len(to_consume) != 0:
            to_write.append(to_consume.pop(0))

        if len(to_write) == shard_sizes[next_output_shard_num]:
            # Write output shard if have number needed.

            if len(to_write) == 0:
                # Insert empty dataframe if nothing to write.
                to_write = [df.iloc[0:0]]

            sh._write_shard(output_sharded, next_output_shard_num,
                            dt.merge_dfs(to_write))
            to_write = []
            next_output_shard_num += 1
            t.update(1)

            if (next_output_shard_num == output_num_shards):
                break
