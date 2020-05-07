"""Code for sharding structures."""
import logging
import os
import shutil

import click
import numpy as np
import pandas as pd
import tqdm

import atom3d.util.datatypes as dt
import atom3d.util.ensemble as en
import atom3d.util.file as fi


@click.command(help='Combine files into sharded HDF5 files.')
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('sharded')
@click.option('--filetype', type=click.Choice(['pdb', 'pdb.gz', 'mmcif']),
              default='pdb', help='which kinds of files are we sharding.')
@click.option('--ensembler', type=click.Choice(en.ensemblers.keys()),
              default='none', help='how to ensemble files')
def shard_dataset(input_dir, sharded, filetype, ensembler):
    """Shard whole input dataset."""
    logging.basicConfig(format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    dirname = os.path.dirname(sharded)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(dirname, exist_ok=True)

    num_shards = get_num_shards(sharded)

    files = fi.find_files(input_dir, dt.patterns[filetype])
    ensembles = en.ensemblers[ensembler](files)

    # Check if already partly written.  If so, resume from there.
    metadata_path = _get_metadata(sharded)
    if os.path.exists(metadata_path):
        metadata = pd.read_hdf(metadata_path, f'metadata')
        num_written = len(metadata['shard_num'].unique())
    else:
        num_written = 0

    shard_ranges = _get_shard_ranges(len(ensembles), num_shards)
    shard_size = shard_ranges[0, 1] - shard_ranges[0, 0]

    total = 0
    logging.info(f'Ensembles per shard: {shard_size:}')
    for shard_num in tqdm.trange(num_written, num_shards):
        start, stop = shard_ranges[shard_num]

        dfs = []
        for name in sorted(ensembles.keys())[start:stop]:
            df = en.parse_ensemble(name, ensembles[name])
            dfs.append(df)
        df = dt.merge_dfs(dfs)

        _write_shard(sharded, shard_num, df)


def read_shard(sharded, shard_num):
    """Read a single shard of a sharded dataset."""
    shard = _get_shard(sharded, shard_num)
    return pd.read_hdf(shard, 'structures')


def read_ensemble(sharded, name):
    """Read ensemble from sharded dataset."""
    metadata_path = _get_metadata(sharded)
    metadata = pd.read_hdf(metadata_path, f'metadata')
    entry = metadata[metadata['ensemble'] == name]
    if len(entry) != 1:
        raise RuntimeError('Need exactly one matchin in structure lookup')
    entry = entry.iloc[0]

    shard = _get_shard(sharded, entry['shard_num'])
    df = pd.read_hdf(shard, 'structures',
                     start=entry['start'], stop=entry['stop'])
    return df.reset_index(drop=True)


def get_names(sharded):
    """Get ensemble names in sharded dataset."""
    metadata_path = _get_metadata(sharded)
    metadata = pd.read_hdf(metadata_path, f'metadata')
    return metadata['name']


def get_num_shards(sharded):
    """Get number of shards in sharded dataset."""
    return int(sharded.split('@')[-1])


def get_num_structures(sharded):
    """Get number of structures in sharded dataset."""
    return get_names(sharded).shape[0]


def move(source_sharded, dest_sharded):
    """Move sharded dataset."""
    copy(source_sharded, dest_sharded)
    delete(source_sharded)


def copy(source_sharded, dest_sharded):
    """Copy sharded dataset."""
    num_shards = get_num_shards(source_sharded)
    for i in range(num_shards):
        source_shard = _get_shard(source_sharded, i)
        dest_shard = _get_shard(dest_sharded, i)
        if os.path.exists(source_shard):
            shutil.copyfile(source_shard, dest_shard)
    source_metadata_path = _get_metadata(source_sharded)
    dest_metadata_path = _get_metadata(dest_sharded)
    if os.path.exists(source_metadata_path):
        shutil.copyfile(source_metadata_path, dest_metadata_path)


def delete(sharded):
    """Delete sharded dataset."""
    num_shards = get_num_shards(sharded)
    for i in range(num_shards):
        shard = _get_shard(sharded, i)
        if os.path.exists(shard):
            os.remove(shard)
    metadata_path = _get_metadata(sharded)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)


def add_to_shard(sharded, shard_num, df, key):
    """Add dataframe under key to shard."""
    shard = _get_shard(sharded, shard_num)
    df.to_hdf(shard, key, mode='a')


def has(sharded, shard_num, key):
    """If key is present in shard."""
    shard = _get_shard(sharded, shard_num)
    with pd.HDFStore(shard, mode='r') as f:
        return key in [x[1:] for x in f.keys()]


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


def _get_shard_ranges(num_structures, num_shards):
    """Get list of shard starts and ends."""
    base_shard_size = int(num_structures / num_shards)
    excess = num_structures - base_shard_size * num_shards
    shard_sizes = np.ones((num_shards), dtype=np.int32) * base_shard_size
    shard_sizes[:excess] += 1
    stops = np.cumsum(shard_sizes)
    starts = stops - shard_sizes
    return np.stack((starts, stops)).T


def _write_shard(sharded, shard_num, df):
    """Write to a single shard of a sharded dataset."""
    metadata = pd.DataFrame(
        [(shard_num, x, y.index[0], y.index[0] + len(y))
         for x, y in dt.split_df(df, 'ensemble')],
        columns=['shard_num', 'ensemble', 'start', 'stop'])

    # Check that we are writing same name again to same sharded dataset.
    metadata_path = _get_metadata(sharded)
    if os.path.exists(metadata_path):
        metadata = pd.concat((pd.read_hdf(metadata_path, f'metadata'),
                              metadata)).reset_index(drop=True)
        if metadata['ensemble'].duplicated().any():
            raise RuntimeError('Writing duplicate to sharded')

    path = _get_shard(sharded, shard_num)
    df.to_hdf(path, f'structures')
    metadata.to_hdf(metadata_path, f'metadata', mode='w')


if __name__ == "__main__":
    shard_dataset()
