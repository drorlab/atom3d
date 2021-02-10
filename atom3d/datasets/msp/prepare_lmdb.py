import numpy as np
import pandas as pd
import collections as col
import logging
import os
import re
import sys
import scipy.spatial
import parallel as par
import click
import torch

sys.path.insert(0, '../../..')
import atom3d.datasets.datasets as da
#import atom3d.datasets.res.util as util
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo

import atom3d.shard.shard as sh

logger = logging.getLogger(__name__)

label_res_dict={0:'HIS',1:'LYS',2:'ARG',3:'ASP',4:'GLU',5:'SER',6:'THR',7:'ASN',8:'GLN',9:'ALA',10:'VAL',11:'LEU',12:'ILE',13:'MET',14:'PHE',15:'TYR',16:'TRP',17:'PRO',18:'GLY',19:'CYS'}
res_label_dict={'HIS':0,'LYS':1,'ARG':2,'ASP':3,'GLU':4,'SER':5,'THR':6,'ASN':7,'GLN':8,'ALA':9,'VAL':10,'LEU':11,'ILE':12,'MET':13,'PHE':14,'TYR':15,'TRP':16,'PRO':17,'GLY':18,'CYS':19}
bb_atoms = ['N', 'CA', 'C', 'O']
allowed_atoms = ['C', 'O', 'N', 'S', 'P', 'SE']

# computed statistics from training set
res_wt_dict = {'HIS': 0.581391659111514, 'LYS': 0.266061611865989, 'ARG': 0.2796785729861747, 'ASP': 0.26563454667840314, 'GLU': 0.22814679094919596, 'SER': 0.2612916369563003, 'THR': 0.27832512315270935, 'ASN': 0.3477441570413752, 'GLN': 0.37781509139381086, 'ALA': 0.20421144813311043, 'VAL': 0.22354397064847012, 'LEU': 0.18395198072344454, 'ILE': 0.2631600545792168, 'MET': 0.6918305148744505, 'PHE': 0.3592224851905275, 'TYR': 0.4048964515721682, 'TRP': 0.9882874205355423, 'PRO': 0.32994186046511625, 'GLY': 0.2238561093317741, 'CYS': 1.0}

gly_CB_mu = np.array([-0.5311191 , -0.75842446,  1.2198311 ], dtype=np.float32)
gly_CB_sigma = np.array([[1.63731114e-03, 2.40018381e-04, 6.38361679e-04],
       [2.40018381e-04, 6.87853419e-05, 1.43898267e-04],
       [6.38361679e-04, 1.43898267e-04, 3.25022011e-04]], dtype=np.float32)


class MSPTransform(object):
    """
    Transform for MSP data.
    """
    def __init__(self, base_file_dir):
        self.mut_file_dir = os.path.join(base_file_dir, 'mutated')
        self.orig_file_dir = os.path.join(base_file_dir, 'original')
        self.labels = self._get_labels_from_file(os.path.join(base_file_dir, 'mutated', 'data_keep.csv'))
        self.mut_orig_mapping = self._match_files(os.listdir(self.mut_file_dir) + os.listdir(self.orig_file_dir))

    def __call__(self, x, error_if_missing=False):
        name = os.path.splitext(x['id'])[0]
        x['id'] = name
        orig_file = self.mut_orig_mapping[name]['original']
        x['original_atoms'] = fo.bp_to_df(fo.read_any(os.path.join(self.orig_file_dir, orig_file)))
        x['mutated_atoms'] = x.pop('atoms')
        x['label'] = str(self.labels.loc[name])
        return x
    
    def _match_files(self, pdb_files):
        """We find matching original pdb for each mutated pdb."""
        # dirs = list(set([os.path.dirname(f) for f in pdb_files]))

        original, mutated = {}, {}
        for f in pdb_files:
            name = os.path.splitext(os.path.basename(f))[0]
            if len(name.split('_')) > 3:
                assert name not in mutated
                mutated[name] = f
            else:
                assert name not in original
                original[name] = f

        ensembles = {}
        for mname, mfile in mutated.items():
            oname = '_'.join(mname.split('_')[:-1])
            ofile = original[oname]
            ensembles[mname] = {
                'original': ofile,
                'mutated': mfile
            }
        return ensembles

    def _get_labels_from_file(self, labels_csv):
        data = pd.read_csv(labels_csv, names=['oname', 'mutation', 'label'])
        data['ensemble'] = data.apply(
            lambda x: x['oname'] + '_' + x['mutation'], axis=1)
        data = data.set_index('ensemble', drop=False)
        return data['label']


@click.command(help='Prepare RES dataset')
@click.argument('input_file_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--split', '-s', is_flag=True)
@click.option('--train_txt', '-tr', type=click.Path(exists=True), default=None)
@click.option('--val_txt', '-v', type=click.Path(exists=True), default=None)
@click.option('--test_txt', '-t', type=click.Path(exists=True), default=None)
def prepare(input_file_path, output_root, split, train_txt, val_txt, test_txt):
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    # Assume PDB filetype.
    filetype = 'pdb'

    file_list = fi.find_files(os.path.join(input_file_path, 'mutated'), fo.patterns[filetype])
    transform = MSPTransform(base_file_dir=input_file_path)
    
    lmdb_path = os.path.join(output_root, 'raw', 'MSP', 'data')
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
        
    logger.info(f'Creating lmdb dataset into {lmdb_path:}...')
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    #dataset = da.load_dataset(file_list, filetype, transform=transform)
    #da.make_lmdb_dataset(dataset, lmdb_path)

    if not split:
        return

    
    logger.info(f'Splitting indices...')
    lmdb_ds = da.load_dataset(lmdb_path, 'lmdb')
    
    split_data_path = os.path.join(output_root, 'splits', 'split-by-seqid30', 'data')
    split_idx_path = os.path.join(output_root, 'splits', 'split-by-seqid30', 'indices')
    if not os.path.exists(split_data_path):
        os.makedirs(split_data_path)
    if not os.path.exists(split_idx_path):
        os.makedirs(split_idx_path)

    def _write_split_indices(split_txt, lmdb_ds, output_txt):
        with open(split_txt, 'r') as f:
            split_set = set([x.strip() for x in f.readlines()])
        # Check if the target in id is in the desired target split set
        split_ids = list(filter(lambda id: id in split_set, lmdb_ds.ids()))
        # Convert ids into lmdb numerical indices and write into txt file
        split_indices = lmdb_ds.ids_to_indices(split_ids)
        str_indices = [str(i) for i in split_indices]
        with open(output_txt, 'w') as f:
            f.write(str('\n'.join(str_indices)))
        return split_indices

    
    logger.info(f'Writing train')
    train_indices = _write_split_indices(train_txt, lmdb_ds, os.path.join(split_idx_path, 'train_indices.txt'))
    print(train_indices)
    train_dataset = torch.utils.data.Subset(lmdb_ds, train_indices)
    da.make_lmdb_dataset(train_dataset, os.path.join(split_data_path, 'train'))

    logger.info(f'Writing val')
    val_indices = _write_split_indices(val_txt, lmdb_ds, os.path.join(split_idx_path, 'val_indices.txt'))
    val_dataset = torch.utils.data.Subset(lmdb_ds, val_indices)
    da.make_lmdb_dataset(val_dataset, os.path.join(split_data_path, 'val'))

    logger.info(f'Writing test')
    test_indices = _write_split_indices(test_txt, lmdb_ds, os.path.join(split_idx_path, 'test_indices.txt'))
    test_dataset = torch.utils.data.Subset(lmdb_ds, test_indices)
    da.make_lmdb_dataset(test_dataset, os.path.join(split_data_path, 'test'))


if __name__ == "__main__":
    # data_dir = '/scratch/users/raphtown/atom3d_mirror/mutation_stability_prediction/'
    # train_sharded = sh.Sharded.load(os.path.join(data_dir, 'split/pairs_train@40'))
    # keys = train_sharded.get_names()
    # keys.to_csv('train.txt',index=False, header=False)
    # val_sharded = sh.Sharded.load(os.path.join(data_dir, 'split/pairs_val@40'))
    # keys = val_sharded.get_names()
    # keys.to_csv('val.txt',index=False, header=False)
    # test_sharded = sh.Sharded.load(os.path.join(data_dir, 'split/pairs_test@40'))
    # keys = test_sharded.get_names()
    # keys.to_csv('test.txt',index=False, header=False)
    # quit()
    prepare()
