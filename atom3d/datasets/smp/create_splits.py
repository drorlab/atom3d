import click
import os
import pandas as pd
import numpy as np
import atom3d.shard.shard as sh
import atom3d.shard.shard_ops as sho
import atom3d.filters.filters as filters

#dirname_all   = 'data/qm9/sharded/qm9_all'
#dirname_split = 'data/qm9/sharded/qm9_split'
#input_ds_name = 'data/qm9/sharded/qm9@1'
#input_splits  = 'data/qm9/splits'
#input_csvfile = 'data/qm9/raw/gdb9_with_cv_atom.csv'
#input_exclude = 'data/qm9/raw/excl.dat'


@click.command(help='Split the sharded QM9 dataset according to an externally defined split.')
@click.argument('dirname_all',   type=click.Path())
@click.argument('dirname_split', type=click.Path())
@click.argument('input_ds_name', type=click.Path())
@click.argument('input_splits',  type=click.Path(exists=True))
@click.argument('input_csvfile', type=click.Path(exists=True))
@click.argument('input_exclude', type=click.Path(exists=True))
#@click.option('-n', '--num_threads', default=8,
#              help='Number of threads to use for parallel processing.')
#@click.option('--overwrite/--no-overwrite', default=False,
#              help='Overwrite existing labels.')


def split_dataset(dirname_all, dirname_split, input_ds_name, input_splits, input_csvfile, input_exclude):
    
    # Read the sharded dataset
    input_sharded = sh.Sharded.load(input_ds_name)
    input_shard = input_sharded.read_shard(0)
    input_label = input_sharded.read_shard(0,'labels')

    # Create output directories
    if not os.path.exists(dirname_all) and dirname_all != '':
        os.makedirs(dirname_all, exist_ok=True)
    if not os.path.exists(dirname_split) and dirname_split != '':
        os.makedirs(dirname_split, exist_ok=True)

    # Correct for ensemble = None
    input_shard['ensemble'] = input_shard['model']

    # Save the full (corrected) dataset
    sharded_all = sh.Sharded(dirname_all+'/qm9_all@1', input_sharded.get_keys())
    sharded_all._write_shard(0, input_shard)
    sharded_all.add_to_shard(0, input_label, 'labels')

    # Read raw and split data
    label_data = pd.read_csv(input_csvfile)
    indices_ex = np.loadtxt(input_exclude, dtype=int)
    indices_tr = np.loadtxt(input_splits+'/indices_train.dat', dtype=int)
    indices_va = np.loadtxt(input_splits+'/indices_valid.dat', dtype=int)
    indices_te = np.loadtxt(input_splits+'/indices_test.dat', dtype=int)
    
    # Create lists of molecule IDs for exclusion and splits
    mol_ids = label_data['mol_id'].tolist()
    mol_ids_ex = label_data.loc[indices_ex]['mol_id'].tolist()
    mol_ids_te = label_data.loc[indices_te]['mol_id'].tolist()
    mol_ids_va = label_data.loc[indices_va]['mol_id'].tolist()
    mol_ids_tr = label_data.loc[indices_tr]['mol_id'].tolist()

    # Write lists of mol_ids to files
    with open(dirname_split+'/mol_ids_excluded.txt', 'w') as f:
        for mol_id in mol_ids_ex: f.write("%s\n" % mol_id)
    with open(dirname_split+'/mol_ids_training.txt', 'w') as f:
        for mol_id in mol_ids_tr: f.write("%s\n" % mol_id)
    with open(dirname_split+'/mol_ids_validation.txt', 'w') as f:
        for mol_id in mol_ids_va: f.write("%s\n" % mol_id)
    with open(dirname_split+'/mol_ids_test.txt', 'w') as f:
        for mol_id in mol_ids_te: f.write("%s\n" % mol_id)
            
    # Split the labels
    labels_te = input_label.loc[label_data['mol_id'].isin(mol_ids_te)].reset_index(drop=True)
    labels_va = input_label.loc[label_data['mol_id'].isin(mol_ids_va)].reset_index(drop=True)
    labels_tr = input_label.loc[label_data['mol_id'].isin(mol_ids_tr)].reset_index(drop=True)
    
    # Filter and write out training set
    filter_tr  = filters.form_filter_against_list(mol_ids_tr, 'subunit')
    sharded_tr = sh.Sharded(dirname_split+'/train@1', sharded_all.get_keys())
    sho.filter_sharded(sharded_all, sharded_tr, filter_tr)
    sharded_tr.add_to_shard(0, labels_tr, 'labels')

    # Filter and write out validation set
    filter_va  = filters.form_filter_against_list(mol_ids_va, 'structure')
    sharded_va = sh.Sharded(dirname_split+'/val@1', sharded_all.get_keys())
    sho.filter_sharded(sharded_all, sharded_va, filter_va)
    sharded_va.add_to_shard(0, labels_va, 'labels')

    # Filter and write out test set
    filter_te  = filters.form_filter_against_list(mol_ids_te, 'structure')
    sharded_te = sh.Sharded(dirname_split+'/test@1', input_sharded.get_keys())
    sho.filter_sharded(sharded_all, sharded_te, filter_te)
    sharded_te.add_to_shard(0, labels_te, 'labels')
    

if __name__ == "__main__":
    split_dataset()
    
    