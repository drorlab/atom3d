import collections as col
import logging
import os
import sys

import click
import pandas as pd

from rdkit import Chem
from torch.utils.data import Dataset

import atom3d.datasets.datasets as da
import atom3d.protein.sequence as seq
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo

logger = logging.getLogger(__name__)


class Scores(object):
    """
    Track and lookup score files.
    """
    def __init__(self, score_path):
        self._scores = pd.read_csv(
            score_path, delimiter=',', engine='python',
            index_col='pdb', usecols=['pdb', 'neglog_aff']).dropna()

    def _lookup(self, pdbcode):
        if pdbcode in self._scores.index:
            return self._scores.loc[pdbcode].to_dict()
        return None

    def __call__(self, x, error_if_missing=False):
        x['scores'] = self._lookup(x['id'])
        if x['scores'] is None and error_if_missing:
            raise RuntimeError(f'Unable to find scores for {x["id"]}')
        return x


class SequenceReader(object):
    """
    Track and lookup protein AA sequences.
    """
    def __init__(self, protein_dir):
        self._protein_dir = protein_dir

    def _lookup(self, file_path):
        return seq.get_chain_sequences(fo.bp_to_df(fo.read_any(file_path)))

    def __call__(self, x, error_if_missing=False):
        x['seq'] = self._lookup(x['file_path'])
        del x['file_path']
        if x['seq'] is None and error_if_missing:
            raise RuntimeError(f'Unable to find AA sequence for {x["id"]}')
        return x


class SmilesReader(object):
    """
    Track and lookup ligand SMILES.
    """
    def _lookup(self, file_path):
        # Assume one ligand in the pdbbind ligand SDF file
        ligand = fo.read_sdf_to_mol(file_path, sanitize=False,
                                    add_hs=False, remove_hs=True)[0]
        return Chem.MolToSmiles(ligand)

    def __call__(self, x, error_if_missing=False):
        x['smiles'] = self._lookup(x['file_path'])
        del x['file_path']
        if x['smiles'] is None and error_if_missing:
            raise RuntimeError(f'Unable to find SMILES for {x["id"]}')
        return x


class LBADataset(Dataset):
    def __init__(self, input_file_path, pdbcodes, transform=None):
        self._protein_dataset = None
        self._pocket_dataset = None
        self._ligand_dataset = None
        self._load_datasets(input_file_path, pdbcodes)

        self._num_examples = len(self._protein_dataset)
        self._transform = transform

    def _load_datasets(self, input_file_path, pdbcodes):
        protein_list = []
        pocket_list = []
        ligand_list = []
        for pdbcode in pdbcodes:
            protein_path = os.path.join(input_file_path, f'{pdbcode:}/{pdbcode:}_protein.cif')
            pocket_path = os.path.join(input_file_path, f'{pdbcode:}/{pdbcode:}_pocket.cif')
            ligand_path = os.path.join(input_file_path, f'{pdbcode:}/{pdbcode:}_ligand.sdf')
            if os.path.exists(protein_path) and os.path.exists(pocket_path) and \
                    os.path.exists(ligand_path):
                protein_list.append(protein_path)
                pocket_list.append(pocket_path)
                ligand_list.append(ligand_path)
        assert len(protein_list) == len(pocket_list) == len(ligand_list)
        logger.info(f'Found {len(protein_list):} protein/ligand files...')

        self._protein_dataset = da.load_dataset(protein_list, 'pdb',
                                                transform=SequenceReader(input_file_path))
        self._pocket_dataset = da.load_dataset(pocket_list, 'pdb',
                                               transform=None)
        self._ligand_dataset = da.load_dataset(ligand_list, 'sdf', include_bonds=True, add_Hs=False,
                                               transform=SmilesReader())

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        protein = self._protein_dataset[index]
        pocket = self._pocket_dataset[index]
        ligand = self._ligand_dataset[index]
        pdbcode = fi.get_pdb_code(protein['id'])

        item = {
            'atoms_protein': protein['atoms'],
            'atoms_pocket': pocket['atoms'],
            'atoms_ligand': ligand['atoms'],
            'bonds': ligand['bonds'],
            'id': pdbcode,
            'seq': protein['seq'],
            'smiles': ligand['smiles'],
        }
        if self._transform:
            item = self._transform(item)
        return item


def split_lmdb_dataset(lmdb_path, train_txt, val_txt, test_txt, split_dir):
    logger.info(f'Splitting indices, load data from {lmdb_path:}...')
    lmdb_ds = da.load_dataset(lmdb_path, 'lmdb')

    def _write_split_indices(split_txt, lmdb_ds, output_txt):
        with open(split_txt, 'r') as f:
            split_set = set([x.strip() for x in f.readlines()])
        # Check if the pdbcode in id is in the desired pdbcode split set
        split_ids = list(filter(lambda id: id in split_set, lmdb_ds.ids()))
        # Convert ids into lmdb numerical indices and write into txt file
        split_indices = lmdb_ds.ids_to_indices(split_ids)
        with open(output_txt, 'w') as f:
            f.write(str('\n'.join([str(i) for i in split_indices])))
        return split_indices

    logger.info(f'Write results to {split_dir:}...')
    os.makedirs(os.path.join(split_dir, 'indices'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'data'), exist_ok=True)

    indices_train = _write_split_indices(
        train_txt, lmdb_ds, os.path.join(split_dir, 'indices/train_indices.txt'))
    indices_val = _write_split_indices(
        val_txt, lmdb_ds, os.path.join(split_dir, 'indices/val_indices.txt'))
    indices_test = _write_split_indices(
        test_txt, lmdb_ds, os.path.join(split_dir, 'indices/test_indices.txt'))

    train_dataset, val_dataset, test_dataset = spl.split(
        lmdb_ds, indices_train, indices_val, indices_test)
    da.make_lmdb_dataset(train_dataset, os.path.join(split_dir, 'data/train'))
    da.make_lmdb_dataset(val_dataset, os.path.join(split_dir, 'data/val'))
    da.make_lmdb_dataset(test_dataset, os.path.join(split_dir, 'data/test'))


def make_lmdb_dataset(input_file_path, score_path, output_root):
    scores = Scores(score_path) if score_path else None

    # Assume subdirectories containing the protein/pocket/ligand files are
    # structured as <input_file_path>/<pdbcode>
    pdbcodes = os.listdir(input_file_path)

    lmdb_path = os.path.join(output_root, 'data')
    os.makedirs(lmdb_path, exist_ok=True)
    logger.info(f'Creating lmdb dataset into {lmdb_path:}...')

    dataset = LBADataset(input_file_path, pdbcodes, transform=scores)
    da.make_lmdb_dataset(dataset, lmdb_path)
    return lmdb_path


@click.command(help='Prepare LBA dataset')
@click.argument('input_file_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--no_gen', '-ng', is_flag=True,
              help="If specified, we don't process the data to lmdb")
@click.option('--split', '-s', is_flag=True)
@click.option('--train_txt', '-tr', type=click.Path(exists=True), default=None)
@click.option('--val_txt', '-v', type=click.Path(exists=True), default=None)
@click.option('--test_txt', '-t', type=click.Path(exists=True), default=None)
@click.option('--score_path', type=click.Path(exists=True), default=None)
def prepare(input_file_path, output_root, no_gen, split,
            train_txt, val_txt, test_txt, score_path):
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    if not no_gen:
        input_file_path = make_lmdb_dataset(input_file_path, score_path, output_root)
        output_root = os.path.join(output_root, 'split')
    if split:
        split_lmdb_dataset(input_file_path, train_txt, val_txt, test_txt, output_root)


if __name__ == "__main__":
    prepare()
