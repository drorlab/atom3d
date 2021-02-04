import contextlib
import gzip
import importlib
import json
import io
import logging
import msgpack
import os
from pathlib import Path
import pickle as pkl
import tqdm
import urllib.request
import subprocess

import Bio.PDB
import lmdb
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, IterableDataset

import atom3d.util.rosetta as ar
import atom3d.util.file as fi
import atom3d.util.formats as fo

logger = logging.getLogger(__name__)


class LMDBDataset(Dataset):
    """
    Creates a dataset from an lmdb file. Adapted from `TAPE <https://github.com/songlab-cal/tape/blob/master/tape/datasets.py>`_.

    :param data_file: path to LMDB file containing dataset
    :type data_file: Union[str, Path]
    """

    def __init__(self, data_file, transform=None):
        """constructor

        """
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
        idx = self.id_to_idx(id)
        return self[idx]

    def id_to_idx(self, id: str):
        if id not in self._id_to_idx:
            raise IndexError(id)
        idx = self._id_to_idx[id]
        return idx

    def ids_to_indices(self, ids):
        return [self.id_to_idx(id) for id in ids]

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
    Creates a dataset from a list of PDB files.

    :param file_list: path to LMDB file containing dataset
    :type file_list: list[Union[str, Path]]
    :param transform: transformation function for data augmentation, defaults to None
    :type transform: function, optional
    """

    def __init__(self, file_list, transform=None, store_file_path=True):
        """constructor

        """
        self._file_list = [Path(x).absolute() for x in file_list]
        self._num_examples = len(self._file_list)
        self._transform = transform
        self._store_file_path = store_file_path

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        file_path = self._file_list[index]

        item = {
            'atoms': fo.bp_to_df(fo.read_any(file_path)),
            'id': file_path.name
        }
        if self._store_file_path:
            item['file_path'] = str(file_path)
        if self._transform:
            item = self._transform(item)
        return item


class SilentDataset(IterableDataset):
    """
    Creates a dataset from rosetta silent files. Can either use a directory of silent files, or a path to one.

    :param file_list: list containing paths to silent files
    :type file_list: list[Union[str, Path]]
    :param transform: transformation function for data augmentation, defaults to None
    :type transform: function, optional
    """

    def __init__(self, file_list, transform=None):
        """constructor

        """

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
        self._scores = ar.Scores(self._file_list)
        self._transform = transform

        self._num_examples = len(self._scores)

    def __len__(self) -> int:
        return len(self._scores)

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
                item['scores'] = self._scores(item)

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
    Creates a dataset from list of XYZ files.

    :param file_list: list containing paths to xyz files
    :type file_list: list[Union[str, Path]]
    :param transform: transformation function for data augmentation, defaults to None
    :type transform: function
    :param gdb: whether to add new energies with subtracted thermochemical energies (for SMP dataset), defaults to False
    :type gdb: bool, optional
    """

    def __init__(self, file_list, transform=None, gdb=False):
        """constructor

        """
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
            item['labels'] = data
            item['freq'] = freq
        if self._transform:
            item = self._transform(item)
        return item


class SDFDataset(Dataset):
    """
    Creates a dataset from directory of SDF files.

    :param file_list: list containing paths to SDF files. Assumes one structure per file.
    :type file_list: list[Union[str, Path]]
    :param transform: transformation function for data augmentation, defaults to None
    :type transform: function, optional
    :param read_bonds: flag for whether to process bond information from SDF, defaults to False
    :type read_bonds: bool, optional
    """

    def __init__(self, file_list, transform=None, read_bonds=False):
        """constructor

        """
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
    """
    Serializes dataset `x` in format given by `serialization_format` (pkl, json, msgpack).
    """
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
    """
    Deserializes dataset `x` assuming format given by `serialization_format` (pkl, json, msgpack).
    """
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
    elif os.path.isfile(input_path):
        with open(input_path) as f:
            all_paths = f.readlines()
        input_dir = os.path.dirname(input_path)
        file_list = []
        for x in all_paths:
            x = x.strip()
            if not fo.is_type(x, filetype):
                continue
            x = os.path.join(input_dir, x)
            file_list.append(x)
    else:
        file_list = fi.find_files(input_path, fo.patterns[filetype])
    return sorted(file_list)


def load_dataset(file_list, filetype, transform=None, include_bonds=False):
    """
    Load files in file_list into corresponding dataset object. All files should be of type filetype.

    :param file_list: List containing paths to files. Assumes one structure per file.
    :type file_list: list[Union[str, Path]]
    :param filetype: Type of dataset. Allowable types are 'lmdb', 'pdb', 'silent', 'sdf', 'xyz', 'xyz-gdb'.
    :type filetype: str
    :param transform: transformation function for data augmentation, defaults to None
    :type transform: function, optional
    :param include_bonds: flag for whether to process bond information for small molecules, defaults to False
    :type include_bonds: bool, optional

    :return: Pytorch Dataset containing data
    :rtype: torch.utils.data.Dataset
    """
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


def make_lmdb_dataset(dataset, output_lmdb,
                      filter_fn=None, serialization_format='json',
                      include_bonds=False):
    """
    Make an LMDB dataset from an input dataset.

    :param dataset: Input dataset to convert
    :type dataset: torch.utils.data.Dataset
    :param output_lmdb: Path to output LMDB.
    :type output_lmdb: Union[str, Path]
    :param filter_fn: Filter to decided if removing files.
    :type filter_fn: lambda x -> True/False
    :param serialization_format: How to serialize an entry.
    :type serialization_format: 'json', 'msgpack', 'pkl'
    :param include_bonds: Include bond information (only available for SDF yet).
    :type include_bonds: bool
    """

    num_examples = len(dataset)

    logger.info(f'{num_examples} examples')

    env = lmdb.open(str(output_lmdb), map_size=int(1e11))

    with env.begin(write=True) as txn:

        id_to_idx = {}
        i = 0
        for x in tqdm.tqdm(dataset, total=num_examples):
            if filter_fn is not None and filter_fn(x):
                continue
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                f.write(serialize(x, serialization_format))
            compressed = buf.getvalue()
            result = txn.put(str(i).encode(), compressed, overwrite=False)
            if not result:
                raise RuntimeError(f'LMDB entry {i} in {str(output_lmdb)} '
                                   'already exists')

            id_to_idx[x['id']] = i
            i += 1

        txn.put(b'num_examples', str(i).encode())
        txn.put(b'serialization_format', serialization_format.encode())
        txn.put(b'id_to_idx', serialize(id_to_idx, serialization_format))


def extract_coordinates_as_numpy_arrays(dataset, indices=None, atom_frames=['atoms'], drop_elements=[]):
    """Convert the molecules from a dataset to a dictionary of numpy arrays.
       Labels are not processed; they are handled differently for every dataset.

    :param dataset: LMDB dataset from which to extract coordinates.
    :type dataset: torch.utils.data.Dataset
    :param indices: Indices of the items for which to extract coordinates.
    :type indices: numpy.array
    :param atom_frames: keys for the frames that contain the atoms to be written.
    :type atom_frames: [str]

    :return: Dictionary of numpy arrays with number of atoms, charges, and positions
    :rtype: dict
    """
    # Size of the dataset
    if indices is None:
        indices = np.arange(len(dataset))
    else:
        assert len(dataset) > max(indices)
    num_items = len(indices)

    # Calculate number of atoms for each molecule
    num_atoms = []
    for idx in indices:
        item = dataset[idx]
        atoms = pd.concat([item[frame] for frame in atom_frames])
        keep = np.array([el not in drop_elements for el in atoms['element']])
        num_atoms.append(sum(keep))

    # All charges and position arrays have the same size
    arr_size  = np.max(num_atoms)
    charges   = np.zeros([num_items,arr_size])
    positions = np.zeros([num_items,arr_size,3])
    # For each molecule and each atom...
    for j,idx in enumerate(indices):
        item = dataset[idx]
        # concatenate atoms from all desired frames
        all_atoms = [item[frame] for frame in atom_frames]
        atoms = pd.concat(all_atoms, ignore_index=True)
        # only keep atoms that are not one of the elements to drop
        keep = np.array([el not in drop_elements for el in atoms['element']])
        atoms_to_keep = atoms[keep].reset_index(drop=True)
        # write per-atom data to arrays
        for ia in range(num_atoms[j]):
            element = atoms_to_keep['element'][ia].title()
            charges[j,ia] = fo.atomic_number[element]
            positions[j,ia,0] = atoms_to_keep['x'][ia]
            positions[j,ia,1] = atoms_to_keep['y'][ia]
            positions[j,ia,2] = atoms_to_keep['z'][ia]

    # Create a dictionary with all the arrays
    numpy_dict = {'index':indices, 'num_atoms':num_atoms,
                  'charges':charges, 'positions':positions}

    return numpy_dict

def combine_datasets(dataset_list, output_lmdb, filter_fn=None, serialization_format='json'):
    """
    Combine list of datasets (in any format) to single LMDB dataset.

    :param dataset_list: List of input datasets
    :type dataset_list: List[torch.utils.data.Dataset]
    :param output_lmdb: Path to output LMDB.
    :type output_lmdb: Union[str, Path]
    :param filter_fn: Filter to decided if removing files.
    :type filter_fn: lambda x -> True/False
    :param serialization_format: How to serialize an entry.
    :type serialization_format: 'json', 'msgpack', 'pkl'
    """

    num_examples = np.sum([len(d) for d in dataset_list])

    logger.info(f'{num_examples} examples in combined dataset')

    env = lmdb.open(str(output_lmdb), map_size=int(1e11))

    with env.begin(write=True) as txn:

        id_to_idx = {}
        i = 0

        for dset in dataset_list:
            for x in tqdm.tqdm(dset, initial=i, total=num_examples):
                if filter_fn is not None and filter_fn(x):
                    continue
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                    f.write(serialize(x, serialization_format))
                compressed = buf.getvalue()
                result = txn.put(str(i).encode(), compressed, overwrite=False)
                if not result:
                    raise RuntimeError(f'LMDB entry {i} in {str(output_lmdb)} '
                                    'already exists')

                id_to_idx[x['id']] = i
                i += 1

        txn.put(b'num_examples', str(i).encode())
        txn.put(b'serialization_format', serialization_format.encode())
        txn.put(b'id_to_idx', serialize(id_to_idx, serialization_format))

def download_dataset(name, out_path):
    """Download an ATOM3D dataset in LMDB format. Available datasets are SMP, PIP, RES, MSP, LBA, LEP, PSR, RSR. Please see `FAQ <datasets target>`_ or `atom3d.ai <atom3d.ai>`_ for more details on each dataset.

    :param name: Three-letter code for dataset (not case-sensitive).
    :type name: str
    :param out_path: Path to directory in which to save downloaded dataset.
    :type out_path: str
    """

    def _hook(t):
        """from https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
        """
        last_b = [0]

        def update_to(b=1, bsize=1, tsize=None):
            """
            b  : int, optional
                Number of blocks transferred so far [default: 1].
            bsize  : int, optional
                Size of each block (in tqdm units) [default: 1].
            tsize  : int, optional
                Total size (in tqdm units). If [default: None] or -1,
                remains unchanged.
            """
            if tsize not in (None, -1):
                t.total = tsize
            displayed = t.update((b - last_b[0]) * bsize)
            last_b[0] = b
            return displayed

        return update_to

    name = name.lower()
    if name == 'smp':
        link = '13MT_f86so0fm6TOtzhW2Qy9ubVQo6UiU'
    elif name == 'pip':
        link = '1D4gMdJEz-6hzSc7_QQ2CF1K-anR4mO8T'
    elif name == 'res':
        link = '1XgZ19YYwloHxEtZUk78PLVzHipFkqIm5'
    elif name == 'msp':
        link = '15rojYF-UjNnqoD8BnNpFtoxVZu64Y7FL'
    elif name == 'lba':
        link = '1CGCRj3IwbT0HNSHIqQ46-o2n1CmGOnwK'
    elif name == 'lep':
        link = '15A85q2h6C1WFKjVttv6sInFNnB5z7Ha7'
    elif name == 'psr':
        link = '1rvxf9JKTq0OvU3QLkxNYomfyXg5sd2CO'
    elif name == 'rsr':
        link = '1rlQ8BmyamMud2TZkcFGy_raz9iI1-KMm'
    else:
        print('Invalid dataset name specified. Possible values are {SMP, PIP, RES, MSP, LBA, LEP, PSR, RSR}')

    # f_out = os.path.join(out_path, name + '.lmdb')
    # with tqdm.tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f_out) as t:  # all optional kwargs
    #     urllib.request.urlretrieve(link, filename=f_out,
    #                     reporthook=_hook(t), data=None)

    cmd = f"wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={link}' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p'  | tr -d \"n\")&id={link}\" -O {name}.tar.gz"
    subprocess.call(cmd, shell=True)
    cmd2 = f"tar xzvf {name}.tar.gz"
    subprocess.call(cmd2, shell=True)

