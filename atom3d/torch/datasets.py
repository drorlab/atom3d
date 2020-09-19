import click
import contextlib
import gzip
import importlib
import json
import io
import logging
import msgpack
from pathlib import Path
import pickle as pkl
import tqdm

import Bio.PDB
import lmdb
import pandas as pd
from torch.utils.data import Dataset, IterableDataset

from . import scores as sc
import atom3d.util.file as fi
import atom3d.util.formats as fo

logger = logging.getLogger(__name__)


class LMDBDataset(Dataset):
    """
    Creates a dataset from an lmdb file.

    Adapted from:
    https://github.com/songlab-cal/tape/blob/master/tape/datasets.py

    Args:
        data_file (Union[str, Path]):
            Path to lmdb file.
    """

    def __init__(self, data_file, transform=None):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b'num_examples'))
            self._serialization_format = \
                txn.get(b'serialization_format').decode()

        self._env = env
        self._transform = transform

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        with self._env.begin(write=False) as txn:

            compressed = txn.get(str(index).encode())
            buf = io.BytesIO(compressed)
            with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                serialized = f.read()
            item = deserialize(serialized, self._serialization_format)

        if 'atoms' in item:
            item['atoms'] = pd.DataFrame(**item['atoms'])

        if self._transform:
            item = self._transform(item)
        if 'id' not in item:
            item['id'] = str(index)
        return item


class PDBDataset(Dataset):
    """
    Creates a dataset from directory of PDB files.

    Args:
        data_path (Union[str, Path, list[str, Path]]):
            Path to pdb files.
    """
    def __init__(self, data_path, transform=None):
        self._file_list = fi.get_file_list(data_path, '.pdb')
        self._num_examples = len(self._file_list)
        self._transform = transform

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        file_path = self._file_list[index]

        item = {
            'atoms': fo.bp_to_df(fo.read_any(file_path)),
            'id': file_path.name,
            'file_path': str(file_path),
        }
        if self._transform:
            item = self._transform(item)
        return item


class SilentDataset(IterableDataset):
    """
    Creates a dataset from rosetta silent files.

    Can either use a directory of silent files, or a path to one.

    Args:
        data_path (Union[str, Path, list[str, Path]]):
            Path to silent files.
    """

    def __init__(self, data_path, transform=None):

        if not importlib.util.find_spec("rosetta") is not None:
            raise RuntimeError(
                'Need to install pyrosetta to process silent files.')

        with contextlib.redirect_stdout(None):
            self.pyrosetta = importlib.import_module('pyrosetta')
            self.pyrpose = importlib.import_module(
                'pyrosetta.rosetta.core.pose')
            self.pyrps = importlib.import_module(
                'pyrosetta.rosetta.core.import_pose.pose_stream')
            self.pyrosetta.init("-mute all")

        self._file_list = fi.get_file_list(data_path, '.out')
        self._num_examples = sum(
            [x.shape[0] for x in self._file_scores.values()])
        self._transform = transform

        self._file_scores = {}
        for silent_file in self._file_list:
            self._file_scores[silent_file] = sc.parse_scores(silent_file)

    def __len__(self) -> int:
        return self._num_examples

    def __iter__(self):
        for silent_file in self._file_list:
            pis = self.pyrps.SilentFilePoseInputStream(str(silent_file))
            while pis.has_another_pose():
                pose = self.pyrosetta.Pose()
                pis.fill_pose(pose)

                item = {
                    'atoms': self._pose_to_df(pose),
                    'id': self.pyrpose.tag_from_pose(pose),
                }
                item['scores'] = \
                    self._file_scores[silent_file].loc[item['id']].to_dict()

                if self._transform:
                    item = self._transform(item)

                yield item

    def _pose_to_df(self, pose):
        """
        Convert pyrosetta representation to pandas dataframe representation.
        """
        name = pose.pdb_info().name()
        string_stream = self.pyrosetta.rosetta.std.ostringstream()
        pose.dump_pdb(string_stream)
        f = io.StringIO(string_stream.str())
        parser = Bio.PDB.PDBParser(QUIET=True)
        bp = parser.get_structure(name, f)
        return fo.bp_to_df(bp)


def serialize(x, serialization_format):
    if serialization_format == 'pkl':
        # Pickle
        # Memory efficient but brittle across languages/python versions.
        return pkl.dumps(x)
    elif serialization_format == 'json':
        # JSON
        # Takes more memory, but widely supported.
        serialized = json.dumps(
            x, default=lambda df: json.loads(
                df.to_json(orient='split', double_precision=6))).encode()
    elif serialization_format == 'msgpack':
        # msgpack
        # A bit more memory efficient than json, a bit less supported.
        serialized = msgpack.packb(
            x, default=lambda df: df.to_dict(orient='split'))
    else:
        raise RuntimeError('Invalid serialization format')
    return serialized


def deserialize(x, serialization_format):
    if serialization_format == 'pkl':
        return pkl.loads(x)
    elif serialization_format == 'json':
        serialized = json.loads(x)
    elif serialization_format == 'msgpack':
        serialized = msgpack.unpackb(x)
    else:
        raise RuntimeError('Invalid serialization format')
    return serialized


def make_lmdb_dataset(input_data_path, output_data_file, filetype,
                      transform=None, serialization_format='json'):
    """
    Make an LMDB dataset from an input dataset.

    Args:
        input_data_path (Union[str, Path, list[str, Path]]):
            Path to input files.
        output_data_file (Union[str, Path]):
            Path to output LMDB.
        filetype ('pdb' or 'silent'):
            Input filetype.
        transform (lambda x -> x):
            Transform to apply before writing out files.
        serialization_format ('json', 'msgpack', 'pkl'):
            How to serialize an entry.
    """
    if filetype == 'pdb':
        file_list = fi.get_file_list(input_data_path, '.pdb')
        dataset = PDBDataset(file_list, transform=transform)
    else:
        file_list = fi.get_file_list(input_data_path, '.out')
        dataset = SilentDataset(file_list, transform=transform)

    num_examples = len(dataset)

    logger.info('making final data set from raw data')
    logger.info(f'{num_examples} examples')

    env = lmdb.open(str(output_data_file), map_size=int(1e11))

    with env.begin(write=True) as txn:
        txn.put(b'num_examples', str(num_examples).encode())
        txn.put(b'serialization_format', serialization_format.encode())

        for i, x in tqdm.tqdm(enumerate(dataset), total=num_examples):
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                f.write(serialize(x, serialization_format))
            compressed = buf.getvalue()
            txn.put(str(i).encode(), compressed)


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_data_file', type=click.Path(exists=False))
@click.option('--filetype', type=click.Choice(['pdb', 'silent']),
              default='pdb')
@click.option('-sf', '--serialization_format',
              type=click.Choice(['msgpack', 'pkl', 'json']),
              default='json')
@click.option('--score_path', type=click.Path(exists=True))
def main(input_data_path, output_data_file, filetype, score_path,
         serialization_format):
    """Script wrapper to make_lmdb_dataset to create LMDB dataset."""
    if score_path and type == 'pdb':
        file_list = fi.get_file_list(input_data_path, '.pdb')
        scores = sc.Scores(score_path)
        new_file_list = scores.remove_missing(file_list)
        logger.info(f'Keeping {len(new_file_list)} / {len(file_list)}')
        input_data_path = new_file_list
    else:
        scores = None
    make_lmdb_dataset(input_data_path, output_data_file, filetype, scores,
                      serialization_format)


if __name__ == '__main__':
    main()
