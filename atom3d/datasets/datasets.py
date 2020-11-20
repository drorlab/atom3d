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
import numpy as np
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
        if type(data_file) is list:
            if len(data_file) != 1:
                raise RuntimeError("Need exactly one filepath for lmdb")
            data_file = data_file[0]

        self.data_file = Path(data_file).absolute()
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        env = lmdb.open(str(self.data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b'num_examples'))
            self._serialization_format = \
                txn.get(b'serialization_format').decode()
            self._id_to_idx = deserialize(
                txn.get(b'id_to_idx'), self._serialization_format)

        self._env = env
        self._transform = transform

    def __len__(self) -> int:
        return self._num_examples

    def get(self, id: str):
        if id not in self._id_to_idx:
            raise IndexError(id)

        idx = self._id_to_idx[id]
        return self[idx]

    def ids(self):
        return list(self._id_to_idx.keys())

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        with self._env.begin(write=False) as txn:

            compressed = txn.get(str(index).encode())
            buf = io.BytesIO(compressed)
            with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                serialized = f.read()
            item = deserialize(serialized, self._serialization_format)

        # Items that start with prefix atoms are assumed to be a dataframe.
        for x in item.keys():
            if x.startswith('atoms'):
                item[x] = pd.DataFrame(**item[x])

        if self._transform:
            item = self._transform(item)
        if 'file_path' not in item:
            item['file_path'] = str(self.data_file)
        if 'id' not in item:
            item['id'] = str(index)
        return item


class PDBDataset(Dataset):
    """
    Creates a dataset from directory of PDB files.

    Args:
        file_list (list[Union[str, Path]]):
            Path to pdb files.
    """

    def __init__(self, file_list, transform=None):
        self._file_list = [Path(x).absolute() for x in file_list]
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
        file_list (list[Union[str, Path]]):
            Path to silent files.
    """

    def __init__(self, file_list, transform=None):

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

        self._file_list = [Path(x).absolute() for x in file_list]
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
                    'file_path': str(silent_file),
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


class XYZDataset(Dataset):
    """
    Creates a dataset from directory of XYZ files.

    Args:
        file_list (list[Union[str, Path]]):
            Path to xyz files.
    """

    def __init__(self, file_list, transform=None, gdb=False):
        self._file_list = [Path(x) for x in file_list]
        self._num_examples = len(self._file_list)
        self._transform = transform
        self._gdb = gdb

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        file_path = self._file_list[index]
        bp = fo.read_xyz(file_path, gdb=self._gdb)
        if self._gdb:
            bp, data, freq, smiles, inchi = bp
        df = fo.bp_to_df(bp)

        item = {
            'atoms': df,
            'id': bp.id,
            'file_path': str(file_path),
        }
        if self._gdb:
            item['labels'] = self.data_with_subtracted_thchem_energy(data, df)
            item['freq'] = freq
        if self._transform:
            item = self._transform(item)
        return item

    def data_with_subtracted_thchem_energy(self, data, df):
        """
        Adds energies with subtracted thermochemical energies to the data list
        We only need this for the QM9 dataset (SMP).
        """

        # per-atom thermochem. energies for U0 [Ha], U [Ha], H [Ha], G [Ha], Cv [cal/(mol*K)]
        # https://figshare.com/articles/dataset/Atomref%3A_Reference_thermochemical_energies_of_H%2C_C%2C_N%2C_O%2C_F_atoms./1057643
        thchem_en = {
            'H': [-0.500273, -0.498857, -0.497912, -0.510927, 2.981],
            'C': [-37.846772, -37.845355, -37.844411, -37.861317, 2.981],
            'N': [-54.583861, -54.582445, -54.581501, -54.598897, 2.981],
            'O': [-75.064579, -75.063163, -75.062219, -75.079532, 2.981],
            'F': [-99.718730, -99.717314, -99.716370, -99.733544, 2.981]}

        # Count occurence of each element in the molecule
        counts = df['element'].value_counts()

        # Calculate and subtract thermochemical energies
        u0_atom = data[10] - np.sum([c * thchem_en[el][0]
                                     for el, c in counts.items()])  # U0
        u_atom = data[11] - np.sum([c * thchem_en[el][1]
                                    for el, c in counts.items()])  # U
        h_atom = data[12] - np.sum([c * thchem_en[el][2]
                                    for el, c in counts.items()])  # H
        g_atom = data[13] - np.sum([c * thchem_en[el][3]
                                    for el, c in counts.items()])  # G
        cv_atom = data[14] - np.sum([c * thchem_en[el][4]
                                     for el, c in counts.items()])  # Cv

        # Append new data
        data += [u0_atom, u_atom, h_atom, g_atom, cv_atom]

        return data


class SDFDataset(Dataset):
    """
    Creates a dataset from directory of SDF files.
    Assumes one structure per file!

    Args:
        file_list (list[Union[str, Path]]):
            Path to sdf files.
    """

    def __init__(self, file_list, transform=None, read_bonds=False):
        self._file_list = [Path(x) for x in file_list]
        self._num_examples = len(self._file_list)
        self._transform = transform
        self._read_bonds = read_bonds

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        # Read biopython structure
        file_path = self._file_list[index]
        structure = fo.read_sdf(str(file_path), sanitize=False,
                                add_hs=False, remove_hs=False)
        # assemble the item (no bonds)
        item = {
            'atoms': fo.bp_to_df(structure),
            'id': structure.id,
            'file_path': str(file_path),
        }
        # Add bonds if included
        if self._read_bonds:
            mol = fo.read_sdf_to_mol(str(file_path), sanitize=False,
                                     add_hs=False, remove_hs=False)
            bonds_df = fo.get_bonds_list_from_mol(mol[0])
            item['bonds'] = bonds_df
        if self._transform:
            item = self._transform(item)
        return item


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


def get_file_list(input_path, filetype):
    if filetype == 'lmdb':
        file_list = [input_path]
    else:
        file_list = fi.find_files(input_path, fo.patterns[filetype])
    return file_list


def load_dataset(file_list, filetype, transform=None, include_bonds=False):
    if type(file_list) != list:
        file_list = get_file_list(file_list, filetype)

    if filetype == 'lmdb':
        dataset = LMDBDataset(file_list, transform=transform)
    elif filetype == 'pdb':
        dataset = PDBDataset(file_list, transform=transform)
    elif filetype == 'silent':
        dataset = SilentDataset(file_list, transform=transform)
    elif filetype == 'sdf':
        # TODO: Make read_bonds parameter part of transform.
        dataset = SDFDataset(file_list, transform=transform,
                             read_bonds=include_bonds)
    elif filetype == 'xyz':
        dataset = XYZDataset(file_list, transform=transform)
    elif filetype == 'xyz-gdb':
        # TODO: Make gdb parameter part of transform.
        dataset = XYZDataset(file_list, transform=transform, gdb=True)
    else:
        raise RuntimeError(f'Unrecognized filetype {filetype}.')
    return dataset


def make_lmdb_dataset(input_file_list, output_lmdb, filetype,
                      transform=None, serialization_format='json',
                      include_bonds=False):
    """
    Make an LMDB dataset from an input dataset.

    Args:
        input_file_list (list[Union[str, Path]])
            Path to input files.
        output_lmdb (Union[str, Path]):
            Path to output LMDB.
        filetype ('pdb', 'silent', 'sdf', 'xyz', or 'xyz-gdb'):
            Input filetype.
        transform (lambda x -> x):
            Transform to apply before writing out files.
        serialization_format ('json', 'msgpack', 'pkl'):
            How to serialize an entry.
        include_bonds (bool):
            Include bond information (only available for SDF yet)
    """
    dataset = load_dataset(
        input_file_list, filetype, transform=transform,
        include_bonds=include_bonds)
    num_examples = len(dataset)

    logger.info('making final data set from raw data')
    logger.info(f'{num_examples} examples')

    env = lmdb.open(str(output_lmdb), map_size=int(1e11))

    with env.begin(write=True) as txn:
        txn.put(b'num_examples', str(num_examples).encode())
        txn.put(b'serialization_format', serialization_format.encode())

        id_to_idx = {}
        for i, x in tqdm.tqdm(enumerate(dataset), total=num_examples):
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                f.write(serialize(x, serialization_format))
            compressed = buf.getvalue()
            txn.put(str(i).encode(), compressed)

            id_to_idx[x['id']] = i
        txn.put(b'id_to_idx', serialize(id_to_idx, serialization_format))
