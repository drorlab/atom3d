import collections as col
import logging
import os
import sys
import click
import pandas as pd
from torch.utils.data import Dataset

import atom3d.datasets.datasets as da
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo


logger = logging.getLogger(__name__)

class LEPDataset(Dataset):
    def __init__(self, input_file_path, id_codes, transform=None):
        #id codes should be in format  Lig3465__6E67__4GBR   'ligand id'__pdb1__pdb2

        self._active_dataset = None
        self._inactive_dataset = None
        self._id_codes = id_codes
        self._load_active_and_inactive_datasets(input_file_path, id_codes)
        self._num_examples = len(self._active_dataset)
        self._transform = transform

    def _load_active_and_inactive_datasets(self, input_file_path, id_codes):
        A_list = [] #active conformations
        I_list = [] #inactive conformations
        for code in id_codes: 
            tokens = code.split('__')
            ligand = tokens[0]
            pdb1 = tokens[1]
            pdb2 = tokens[2]
            A_path = os.path.join(input_file_path, f'{ligand}_to_{pdb1}.pdb')
            I_path = os.path.join(input_file_path, f'{ligand}_to_{pdb2}.pdb')
            
            if os.path.exists(A_path) and os.path.exists(I_path):
                A_list.append(A_path)
                I_list.append(I_path)

        assert len(A_list) == len(I_list)
        logger.info(f'Found {len(A_list):} pairs of protein files...')
        self._active_dataset = da.load_dataset(A_list, 'pdb')
        self._inactive_dataset = da.load_dataset(I_list, 'pdb')

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        active = self._active_dataset[index]
        inactive = self._inactive_dataset[index]

        item = {
            'atoms_active': active['atoms'],
            'atoms_inactive': inactive['atoms'],
            'id': self._id_codes[index],
        }
        if self._transform:
            item = self._transform(item)
        return item


class AddInfo(object):
    """
    Track and lookup score files.
    """
    def __init__(self, info_file):
        self._info = pd.read_csv(info_file, delimiter=',', engine='python')

    def _lookup(self, key, lig, act, ina):
        info = self._info
        a = info[info['active_struc'] == lig+'_to_'+act]
        i = a[a['inactive_struc'] == lig+'_to_'+ina]
        v = list(i[key])[0]
        return v

    def __call__(self, x):
        ligand, astruc, istruc = x['id'].split('__')
        #print('Looking up info for '+x['id'])
        for key in ['protein', 'ligand', 'ligand_name', 
                    'active_struc', 'inactive_struc',
                    'label', 'Dgscore', 'gscoreA', 'gscoreI',
                    'pharm_id', 'SMILES']:
            x[key] = self._lookup(key, ligand, astruc, istruc)
        return x


def load_info_csv(info_csv):
    info = pd.read_csv(info_csv)
    info['ensemble'] = info.apply(
        lambda x: x['ligand'] + '__' + x['active_struc'].split('_')[2] + '__' +
        x['inactive_struc'].split('_')[2], axis=1)
    info = info.set_index('ensemble', drop=False)
    # Remove duplicate ensembles.
    info = info[~info.index.duplicated()]
    return info


def split_lmdb_dataset(lmdb_path, train_txt, val_txt, test_txt, split_dir):
    logger.info(f'Splitting indices, load data from {lmdb_path:}...')
    lmdb_ds = da.load_dataset(lmdb_path, 'lmdb')

    def _write_split_indices(split_txt, lmdb_ds, output_txt):
        with open(split_txt, 'r') as f:
            split_set = set([x.strip().split(',')[-1] for x in f.readlines()])
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



def make_lmdb_dataset(input_file_path, csv_file_path, output_root):
    # Assume PDB filetype.
    filetype = 'pdb'
    info = load_info_csv(csv_file_path)
    print(info.head())
    dataset = LEPDataset(input_file_path, info['ensemble'], transform=AddInfo(csv_file_path))
    #create the all dataset (no splitting)
    lmdb_path = os.path.join(output_root, 'all')
    logger.info(f'Creating lmdb dataset into {lmdb_path:}...')
    da.make_lmdb_dataset(dataset, lmdb_path)
    return lmdb_path


@click.command(help='Prepare lep dataset')
@click.argument('csv_file_path', type=click.Path())
@click.argument('input_file_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--split', '-s', is_flag=True)
@click.option('--train_txt', '-tr', type=click.Path(exists=True), default=None)
@click.option('--val_txt', '-va', type=click.Path(exists=True), default=None)
@click.option('--test_txt', '-te', type=click.Path(exists=True), default=None)
def prepare(csv_file_path, input_file_path, output_root, split, train_txt, val_txt, test_txt):
    
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    input_file_path = make_lmdb_dataset(input_file_path, csv_file_path, output_root)
    if split:
        output_root = os.path.join(output_root, 'split')
        split_lmdb_dataset(input_file_path, train_txt, val_txt, test_txt, output_root)

if __name__ == "__main__":
    prepare()
