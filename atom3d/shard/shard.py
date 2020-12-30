"""Code for sharding structures."""
import logging
import multiprocessing as mp
import os
import shutil

import click
import numpy as np
import pandas as pd
import tqdm

import atom3d.shard.ensemble as en
import atom3d.util.file as fi
import atom3d.util.formats as dt

db_sem = mp.Semaphore()


@click.command(help='Combine files into sharded HDF5 files.')
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('sharded_path')
@click.option('--filetype', type=click.Choice(['pdb', 'pdb.gz', 'mmcif', 'sdf']),
              default='pdb', help='which kinds of files are we sharding.')
@click.option('--ensembler', type=click.Choice(en.ensemblers.keys()),
              default='none', help='how to ensemble files')
def shard_dataset(input_dir, sharded_path, filetype, ensembler):
    """Shard whole input dataset."""
    logging.basicConfig(format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    dirname = os.path.dirname(sharded_path)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(dirname, exist_ok=True)

    files = fi.find_files(input_dir, dt.patterns[filetype])
    ensemble_map = en.ensemblers[ensembler](files)
    Sharded.create_from_ensemble_map(ensemble_map, sharded_path)


class Sharded(object):
    """Sharded pandas dataframe representation."""

    def __init__(self, path, keys):
        self.path = path
        self._keys = keys

    @classmethod
    def create_from_ensemble_map(cls, ensemble_map, path):
        sharded = cls(path, ['ensemble'])

        num_shards = sharded.get_num_shards()

        # Check if already partly written.  If so, resume from there.
        metadata_path = sharded._get_metadata()
        if os.path.exists(metadata_path):
            metadata = pd.read_hdf(metadata_path, f'atom3d/data/metadata')
            num_written = len(metadata['shard_num'].unique())
        else:
            num_written = 0

        shard_ranges = _get_shard_ranges(len(ensemble_map), num_shards)
        shard_size = shard_ranges[0, 1] - shard_ranges[0, 0]

        logging.info(f'Ensembles per shard: {shard_size:}')
        for shard_num in tqdm.trange(num_written, num_shards):
            start, stop = shard_ranges[shard_num]

            dfs = []
            for name in sorted(ensemble_map.keys())[start:stop]:
                df = en.parse_ensemble(name, ensemble_map[name])
                dfs.append(df)
            df = dt.merge_dfs(dfs)

            sharded._write_shard(shard_num, df)

    @classmethod
    def load(cls, path):
        """Load a fully written sharded dataset."""

        # Get keys from metadata file.
        sharded = cls(path, None)
        metadata_path = sharded._get_metadata()
        if not os.path.exists(metadata_path):
            raise RuntimeError(f'Metadata for {path:} does not exist')
        metadata = pd.read_hdf(metadata_path, f'metadata')
        keys = metadata.columns.tolist()
        keys.remove('shard_num')
        keys.remove('start')
        keys.remove('stop')

        sharded = cls(path, keys)
        if not sharded.is_written():
            raise RuntimeError(
                f'Sharded loaded from {path:} not fully written.')
        return sharded

    def is_written(self):
        """Check if metadata files and all data files are there."""
        metadata_path = self._get_metadata()
        if not os.path.exists(metadata_path):
            return False
        num_shards = self.get_num_shards()
        for i in range(num_shards):
            shard = self._get_shard(i)
            if not os.path.exists(shard):
                return False
        return True

    def get_keys(self):
        return self._keys

    def iter_shards(self):
        """Iterate through shards."""
        num_shards = self.get_num_shards()
        for i in range(num_shards):
            yield i, self.read_shard(i)

    def read_shard(self, shard_num, key='structures'):
        """Read a single shard of a sharded dataset."""
        shard = self._get_shard(shard_num)
        return pd.read_hdf(shard, key)

    def read_keyed(self, name):
        """Read keyed entry from sharded dataset."""
        metadata_path = self._get_metadata()
        metadata = pd.read_hdf(metadata_path, f'metadata')
        entry = metadata[(metadata[self._keys] == name).any(axis=1)]
        if len(entry) != 1:
            raise RuntimeError('Need exactly one matchin in structure lookup')
        entry = entry.iloc[0]

        shard = self._get_shard(entry['shard_num'])
        df = pd.read_hdf(shard, 'structures',
                         start=entry['start'], stop=entry['stop'])
        return df.reset_index(drop=True)

    def get_names(self):
        """Get keyed names in sharded dataset."""
        metadata_path = self._get_metadata()
        metadata = pd.read_hdf(metadata_path, f'metadata')
        return metadata[self._keys]

    def get_num_keyed(self):
        """Get number of keyed examples in sharded dataset."""
        return self.get_names().shape[0]

    def get_num_structures(self, keys):
        """Get number of structures in sharded dataset."""
        num_structs = 0
        for _, df in self.iter_shards():
            num_structs += df.groupby(keys).ngroups
        return num_structs

    def move(self, dest_path):
        """Move sharded dataset."""
        self.copy(dest_path)
        self.delete_files()
        self.path = dest_path

    def copy(self, dest_path):
        """Copy sharded dataset."""
        num_shards = self.get_num_shards()

        dest_sharded = Sharded(dest_path, self._keys)

        for i in range(num_shards):
            source_shard = self._get_shard(i)
            dest_shard = dest_sharded._get_shard(i)
            if os.path.exists(source_shard):
                shutil.copyfile(source_shard, dest_shard)
        metadata_path = self._get_metadata()
        dest_metadata_path = dest_sharded._get_metadata()
        if os.path.exists(metadata_path):
            shutil.copyfile(metadata_path, dest_metadata_path)
        return dest_sharded

    def delete_files(self):
        """Delete sharded dataset."""
        num_shards = self.get_num_shards()
        for i in range(num_shards):
            shard = self._get_shard(i)
            if os.path.exists(shard):
                os.remove(shard)
        metadata_path = self._get_metadata()
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

    def add_to_shard(self, shard_num, df, key):
        """Add dataframe under key to shard."""
        shard = self._get_shard(shard_num)
        df.to_hdf(shard, key, mode='a')

    def has(self, shard_num, key):
        """If key is present in shard."""
        shard = self._get_shard(shard_num)
        with pd.HDFStore(shard, mode='r') as f:
            return key in [x[1:] for x in f.keys()]

    def get_num_shards(self):
        """Get number of shards in sharded dataset."""
        return get_num_shards(self.path)

    def get_prefix(self):
        return '@'.join(self.path.split('@')[:-1])

    def _get_shard(self, shard_num):
        num_shards = self.get_num_shards()
        prefix = get_prefix(self.path)
        return f'{prefix:}_{shard_num:04d}_{num_shards:}.h5'

    def _get_metadata(self):
        num_shards = self.get_num_shards()
        prefix = get_prefix(self.path)
        return f'{prefix:}_meta_{num_shards:}.h5'

    def _write_shard(self, shard_num, df):
        """Write to a single shard of a sharded dataset."""

        if len(self._keys) == 1:
            metadata = pd.DataFrame(
                [(shard_num, y.index[0], y.index[0] + len(y)) + (x,)
                 for x, y in dt.split_df(df, self._keys)],
                columns=['shard_num', 'start', 'stop'] + self._keys)
        else:
            metadata = pd.DataFrame(
                [(shard_num, y.index[0], y.index[0] + len(y)) + x
                 for x, y in dt.split_df(df, self._keys)],
                columns=['shard_num', 'start', 'stop'] + self._keys)

        path = self._get_shard(shard_num)
        df.to_hdf(path, f'structures')
        with db_sem:
            # Check that we are writing same name again to same sharded
            # dataset.
            metadata_path = self._get_metadata()
            if os.path.exists(metadata_path):
                metadata = pd.concat((pd.read_hdf(metadata_path, f'metadata'),
                                      metadata)).reset_index(drop=True)
                if metadata[self._keys].duplicated().any():
                    raise RuntimeError(f'Writing duplicate to sharded {path:}')

            metadata.to_hdf(metadata_path, f'metadata', mode='w')


def get_prefix(path):
    return '@'.join(path.split('@')[:-1])


def get_num_shards(path):
    """Get number of shards in sharded dataset."""
    return int(path.split('@')[-1])


def _get_shard_ranges(num_structures, num_shards):
    """Get list of shard starts and ends."""
    base_shard_size = int(num_structures / num_shards)
    excess = num_structures - base_shard_size * num_shards
    shard_sizes = np.ones(num_shards, dtype=np.int32) * base_shard_size
    shard_sizes[:excess] += 1
    stops = np.cumsum(shard_sizes)
    starts = stops - shard_sizes
    return np.stack((starts, stops)).T


if __name__ == "__main__":
    shard_dataset()
