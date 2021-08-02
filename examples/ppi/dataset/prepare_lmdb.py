import collections as col
import logging
import os
import sys

import click
import pandas as pd

from torch.utils.data import Dataset, IterableDataset

import atom3d.datasets.datasets as da
import atom3d.splits.splits as spl
import atom3d.util.ensemble as en
import atom3d.util.file as fi
import atom3d.util.formats as fo
import neighbors as nb
import pairs as pr

logger = logging.getLogger(__name__)


class EnsembleDataset(Dataset):
    def __init__(self, file_list, ensembler):
        self._ensemble_map = en.ensemblers[ensembler](file_list)
        self._ensemble_keys = list(self._ensemble_map)
        self._num_examples = len(self._ensemble_map)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        name = self._ensemble_keys[index]
        df = en.parse_ensemble(name, self._ensemble_map[name])
        item = {
            'pairs': df,
            'id': name,
        }
        return item


class PPIDataset(IterableDataset):
    def __init__(self, file_list, cutoff, cutoff_type, ensembler):
        self._ensemble_dataset = EnsembleDataset(file_list, ensembler)
        self._cutoff = cutoff
        self._cutoff_type = cutoff_type
        self._ensembler = ensembler

    def __len__(self) -> int:
        return len(self._ensemble_dataset)

    def __iter__(self):
        for index in range(len(self._ensemble_dataset)):
            struct = self._ensemble_dataset[index]

            if self._ensembler != 'db5':
                ### Extract pairs of chains in contact
                struct['pairs'] = pr._gen_pairs_per_ensemble(
                    struct['pairs'], self._cutoff, self._cutoff_type)
                if len(struct['pairs']) == 0:
                    # Sometimes there is only one chain, and we end up with no pairs
                    continue
                struct['pairs'] = pd.concat(struct['pairs']).reset_index(drop=True)

            ### Extract pairs of amino acids within pair of chains in contact
            for e, ensemble in struct['pairs'].groupby('ensemble'):
                neighbors = nb.neighbors_from_ensemble(
                    ensemble, self._cutoff, self._cutoff_type).reset_index(drop=True)
                item = {
                    'atoms_pairs': ensemble,
                    'atoms_neighbors': neighbors,
                    'id': e,
                }
                yield item


def split_lmdb_dataset(lmdb_path, train_txt, val_txt, test_txt, split_dir):
    logger.info(f'Splitting indices, load data from {lmdb_path:}...')
    lmdb_ds = da.load_dataset(lmdb_path, 'lmdb')

    def _write_split_indices(split_txt, lmdb_ds, output_txt):
        with open(split_txt, 'r') as f:
            split_set = set([x.strip() for x in f.readlines()])

        # Check if the target in id is in the desired target split set
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


def make_lmdb_dataset(input_file_path, filetype, cutoff, cutoff_type,
                      ensembler, output_root):
    lmdb_path = os.path.join(output_root, 'data')
    os.makedirs(lmdb_path, exist_ok=True)
    logger.info(f'Creating lmdb dataset into {lmdb_path:}...')

    file_list = fi.find_files(input_file_path, fo.patterns[filetype])
    logger.info(f'Found {len(file_list):} pdb files to process...')

    dataset = PPIDataset(file_list, cutoff, cutoff_type, ensembler)
    da.make_lmdb_dataset(dataset, lmdb_path)
    return lmdb_path


@click.command(help='Prepare PPI dataset')
@click.argument('input_file_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--no_gen', '-ng', is_flag=True,
              help="If specified, we don't process the data to lmdb")
@click.option('--split', '-s', is_flag=True)
@click.option('--train_txt', '-tr', type=click.Path(exists=True), default=None)
@click.option('--val_txt', '-v', type=click.Path(exists=True), default=None)
@click.option('--test_txt', '-t', type=click.Path(exists=True), default=None)
@click.option('--filetype', type=click.Choice(['pdb', 'pdb.gz', 'mmcif']),
              default='pdb', help='which kinds of files are we sharding.')
@click.option('-c', '--cutoff', type=int, default=8,
              help='Maximum distance (in angstroms), for two residues to be '
              'considered neighbors.')
@click.option('--cutoff-type', default='CA',
              type=click.Choice(['heavy', 'CA'], case_sensitive=False),
              help='How to compute distance between residues: CA is based on '
              'alpha-carbons, heavy is based on any heavy atom.')
@click.option('--ensembler', type=click.Choice(en.ensemblers.keys()),
              default='none', help='how to ensemble files')
def prepare(input_file_path, output_root, no_gen, split, train_txt, val_txt,
            test_txt, filetype, cutoff, cutoff_type, ensembler):
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    if not no_gen:
        input_file_path = make_lmdb_dataset(
            input_file_path, filetype, cutoff, cutoff_type, ensembler, output_root)
        output_root = os.path.join(output_root, 'split')
    if split:
        split_lmdb_dataset(input_file_path, train_txt, val_txt, test_txt, output_root)


if __name__ == "__main__":
    prepare()
