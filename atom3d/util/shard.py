"""Code for sharding structures."""
import logging
import math
import os

import click
import pandas as pd
import tqdm

import atom3d.util.datatypes as dt
import atom3d.util.file as fi


@click.command(help='Combine files into sharded HDF5 files.')
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_sharded')
@click.option('--filetype', type=click.Choice(['pdb', 'pdb.gz', 'mmcif']),
              default='pdb', help='which kinds of files are we sharding.')
@click.option('--onemodel', default=True, help='keep only first model')
def shard_dataset(input_dir, output_sharded, filetype, onemodel):
    """Shard whole input dataset."""
    logging.basicConfig(format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    if not os.path.exists(os.path.dirname(output_sharded)):
        os.makedirs(os.path.dirname(output_sharded))

    num_shards = get_num_shards(output_sharded)

    files = fi.find_files(input_dir, dt.patterns[filetype])

    shard_size = int(math.ceil((1.0 * len(files)) / num_shards))
    logging.info(f'Structures per shard: {shard_size:}')
    for shard_num in tqdm.trange(num_shards):
        start = shard_num * shard_size
        end = min((shard_num + 1) * shard_size, len(files))

        dfs = []
        for f in files[start:end]:
            df = dt.bp_to_df(dt.read_any(f))
            if onemodel:
                df = df[df['model'] == df['model'].unique()[0]]
            dfs.append(df)

        write_shard(dfs, output_sharded, shard_num)


def read_shard(sharded, shard_num):
    """Read a single shard of a sharded dataset."""
    shard = _get_shard(sharded, shard_num)
    raw = pd.read_hdf(shard, 'table')
    dfs = [x for _, x in raw.groupby('structure')]
    return dfs


def write_shard(dfs, sharded, shard_num):
    """Write to a single shard of a sharded dataset."""
    df = pd.concat(dfs)
    path = _get_shard(sharded, shard_num)
    df.to_hdf(path, f'table')


def get_num_shards(sharded):
    """Get number of shards in sharded dataset."""
    return int(sharded.split('@')[-1])


def _get_prefix(sharded):
    return '@'.join(sharded.split('@')[:-1])


def _get_shard(sharded, shard_num):
    num_shards = get_num_shards(sharded)
    prefix = _get_prefix(sharded)
    return f'{prefix:}_{shard_num:04d}_{num_shards:}.h5'


if __name__ == "__main__":
    shard_dataset()
