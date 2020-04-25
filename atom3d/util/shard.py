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
@click.argument('sharded')
@click.option('--filetype', type=click.Choice(['pdb', 'pdb.gz', 'mmcif']),
              default='pdb', help='which kinds of files are we sharding.')
def shard_dataset(input_dir, sharded, filetype):
    """Shard whole input dataset."""
    logging.basicConfig(format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    if not os.path.exists(os.path.dirname(sharded)):
        os.makedirs(os.path.dirname(sharded))

    num_shards = get_num_shards(sharded)

    files = fi.find_files(input_dir, dt.patterns[filetype])

    # Check if already partly written.  If so, resume from there.
    metadata_path = _get_metadata(sharded)
    if os.path.exists(metadata_path):
        metadata = pd.read_hdf(metadata_path, f'metadata')
        num_written = len(metadata['shard_num'].unique())
    else:
        num_written = 0

    shard_size = int(math.ceil((1.0 * len(files)) / num_shards))
    logging.info(f'Structures per shard: {shard_size:}')
    for shard_num in tqdm.trange(num_written, num_shards):
        start = shard_num * shard_size
        stop = min((shard_num + 1) * shard_size, len(files))

        dfs = []
        for f in files[start:stop]:
            df = dt.bp_to_df(dt.read_any(f))
            dfs.append(df)

        _write_shard(sharded, shard_num, dfs)


def read_shard(sharded, shard_num):
    """Read a single shard of a sharded dataset."""
    shard = _get_shard(sharded, shard_num)
    raw = pd.read_hdf(shard, 'structures')
    dfs = [x for _, x in raw.groupby('structure')]
    return dfs


def read_structure(sharded, name):
    """Read structure from sharded dataset."""
    metadata_path = _get_metadata(sharded)
    metadata = pd.read_hdf(metadata_path, f'metadata')
    entry = metadata[metadata['name'] == name]
    if len(entry) != 1:
        raise RuntimeError('Need exactly one matchin in structure lookup')
    entry = entry.iloc[0]

    shard = _get_shard(sharded, entry['shard_num'])
    df = pd.read_hdf(shard, 'structures',
                     start=entry['start'], stop=entry['stop'])
    return df.reset_index(drop=True)


def get_names(sharded):
    """Get structure names in sharded dataset."""
    metadata_path = _get_metadata(sharded)
    metadata = pd.read_hdf(metadata_path, f'metadata')
    return metadata['name']


def get_num_shards(sharded):
    """Get number of shards in sharded dataset."""
    return int(sharded.split('@')[-1])


def get_num_structures(sharded):
    """Get number of structures in sharded dataset."""
    return get_names(sharded).shape[0]


def _get_prefix(sharded):
    return '@'.join(sharded.split('@')[:-1])


def _get_shard(sharded, shard_num):
    num_shards = get_num_shards(sharded)
    prefix = _get_prefix(sharded)
    return f'{prefix:}_{shard_num:04d}_{num_shards:}.h5'


def _get_metadata(sharded):
    num_shards = get_num_shards(sharded)
    prefix = _get_prefix(sharded)
    return f'{prefix:}_meta_{num_shards:}.h5'


def _write_shard(sharded, shard_num, dfs):
    """Write to a single shard of a sharded dataset."""
    df = pd.concat(dfs).reset_index(drop=True)
    metadata = pd.DataFrame(
        [(shard_num, x, y.index[0], y.index[0] + len(y))
         for x, y in df.groupby('structure')],
        columns=['shard_num', 'name', 'start', 'stop'])

    # Check that we are writing same name again to same sharded dataset.
    metadata_path = _get_metadata(sharded)
    if os.path.exists(metadata_path):
        metadata = pd.concat((pd.read_hdf(metadata_path, f'metadata'),
                              metadata)).reset_index(drop=True)
        if metadata['name'].duplicated().any():
            raise RuntimeError('Writing duplicate to sharded')

    path = _get_shard(sharded, shard_num)
    df.to_hdf(path, f'structures')
    metadata.to_hdf(metadata_path, f'metadata', mode='w')


if __name__ == "__main__":
    shard_dataset()
