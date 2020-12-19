import json
import os
import pathlib
import pickle

import numpy as np
import pandas as pd

import dotenv as de
de.load_dotenv(de.find_dotenv(usecwd=True))

from collections import OrderedDict
from Bio.PDB.Polypeptide import PPBuilder
from rdkit import Chem

import atom3d.shard.shard as sh
import atom3d.util.file as fi
import atom3d.util.formats as ft
import atom3d.protein.sequence as ps


###############################################################################
### Protein sequence
###############################################################################

def __get_subunit_name(subunits, mode='inactive'):
    assert len(subunits) == 2
    for name in subunits:
        if name.endswith('_' + mode):
            return name
    return ''


def process_sharded_datasets(info_df, sharded_datasets_dict, all_data_outfile=None):
    info_df = info_df.set_index(['ligand', 'active_struc'])

    all_data_dict = {}
    all_labels = []
    for (dataset_type, sharded_paths) in sharded_datasets_dict.items():
        data_frames = []
        for path in sharded_paths:
            data_frames.append(process_sharded_single(info_df, sh.Sharded.load(path)))
        data_df = pd.concat(data_frames, ignore_index=True)
        data_df = data_df.sort_values(by=['ligand', 'protein']).reset_index(drop=True)
        all_data_dict[dataset_type] = data_df

    all_data_df = pd.concat(all_data_dict.values(), ignore_index=True)
    all_data_df = all_data_df.sort_values(by=['ligand', 'protein']).reset_index(drop=True)

    dataset_splits = {}
    for k, df in all_data_dict.items():
        dataset_splits[k] = list(df[['ligand', 'protein']].itertuples(index=False))

    if all_data_outfile is not None:
        all_data_df.to_pickle(all_data_outfile)
    return all_data_df, dataset_splits


def process_sharded_single(info_df, sharded):
    data = []
    for _, shard_df in sharded.iter_shards():
        for ensemble_name, ensemble_df in shard_df.groupby(['ensemble']):
            active_name = __get_subunit_name(ensemble_df.subunit.unique(), mode='active')
            struct_df = ensemble_df[ensemble_df.subunit == active_name]
            protein_df = struct_df[struct_df.chain != 'L']

            ligand_name = ensemble_name.split('_')[0]
            info = info_df.loc[(ligand_name, protein_df.structure.unique()[0].split('.')[0])]

            # Sequence
            chain_sequences = ps.get_all_chain_sequences_df(protein_df)
            assert len(chain_sequences) == 1
            seq = []
            for (_, s) in chain_sequences[0][1]:
                seq.append(s)
            seq = '\n'.join(seq)

            data.append({
                'ligand': ligand_name,
                'protein': info.protein,
                'label': (info.label == 'A'),
                'smiles': info.SMILES,
                'seq': seq,
            })

    data_df = pd.DataFrame(data, columns=['ligand', 'protein', 'label', 'smiles', 'seq'])
    return data_df


def get_seq_length_stats(data_df):
    lengths_df = data_df.set_index('protein').seq.apply(len).reset_index(
        name='length')
    print(lengths_df.describe())
    return lengths_df


###############################################################################
### Binding affinity matrix
###############################################################################
def gen_binding_aff_matrix(smiles_df, seqs_df, labels_df):
    """
    Binding affinity matrix (drugs x proteins). Each drug-protein pair
    with unknown affinity is indicated as 'nan'.
    """
    aff_matrix = np.full([len(smiles_df), len(seqs_df)], np.nan)
    for _, (ligand, protein, label) in labels_df.iterrows():
        smiles_idx = smiles_df[smiles_df.ligand == ligand].index[0]
        seq_idx = seqs_df[seqs_df.protein == protein].index[0]
        aff_matrix[smiles_idx][seq_idx] = label
    return aff_matrix


###############################################################################
### Helper functions to convert split file into DeepDTA format
###############################################################################

def process_data_to_DeepDTA(all_data_df):
    def __slice(df, col1, col2):
        return df[[col1, col2]].drop_duplicates(subset=[col1]).sort_values(
            by=col1).reset_index(drop=True)

    smiles_df = __slice(all_data_df, 'ligand', 'smiles')
    seqs_df = __slice(all_data_df, 'protein', 'seq')
    labels_df = all_data_df[['ligand', 'protein', 'label']]
    aff_matrix = gen_binding_aff_matrix(smiles_df, seqs_df, labels_df)
    return smiles_df, seqs_df, aff_matrix


def write_DeepDTA_to_files(smiles_df, data_df, aff_matrix,
                           ligands_file, proteins_file, Y_file):
    with open(ligands_file, 'w') as f:
        ordered_dict = smiles_df.set_index('ligand').iloc[:,0].to_dict(into=OrderedDict)
        json.dump(ordered_dict, f)
    with open(proteins_file, 'w') as f:
        ordered_dict = data_df.set_index('protein').iloc[:,0].to_dict(into=OrderedDict)
        json.dump(ordered_dict, f)
    with open(Y_file, 'wb') as f:
        pickle.dump(aff_matrix, f)


def gen_DeepDTA_train_test_folds(smiles_df, seqs_df, aff_matrix,
                                 train_split, test_split,
                                 train_nfold=5, train_fold_file=None,
                                 test_fold_file=None, val_split=[]):
    # row: ligand, col: protein
    label_row_inds, label_col_inds = np.where(np.isnan(aff_matrix)==False)
    train_set, val_set, test_set = [], [], []

    for i, (lig_idx, prot_idx) in enumerate(zip(label_row_inds, label_col_inds)):
        ligand = smiles_df.iloc[lig_idx].ligand
        protein = seqs_df.iloc[prot_idx].protein

        if (ligand, protein) in train_split:
            train_set.append(i)
        elif (ligand, protein) in test_split:
            test_set.append(i)
        elif (ligand, protein) in val_split:
            val_set.append(i)

    if len(val_split) == 0:
        # Split train set into n-folds
        np.random.shuffle(train_set)
        train_folds = np.array_split(train_set, train_nfold)
        train_folds = [arr.tolist() for arr in train_folds]
    else:
        train_folds = [val_set, train_set]
    # Shuffle the folds
    np.random.shuffle(train_set)
    for arr in train_folds:
        np.random.shuffle(arr)

    # Write into files if applicable
    if train_fold_file is not None:
        with open(train_fold_file, 'w') as f:
            f.write(json.dumps(train_folds))
    if test_fold_file is not None:
        with open(test_fold_file, 'w') as f:
            f.write(json.dumps(test_set))
    return train_folds, test_set


if __name__ == '__main__':
    LEP_ROOTDIR = '/oak/stanford/groups/rondror/projects/atom3d/supporting_files/ligand_efficacy_prediction'
    info_file = os.path.join(LEP_ROOTDIR, 'info.csv')

    sharded_datasets_dict = {
        'train': [os.environ['ACTIVITY_TRAIN_SHARDED']],
        'val': [os.environ['ACTIVITY_VAL_SHARDED']],
        'test': [os.environ['ACTIVITY_TEST_SHARDED']],
    }

    DEEP_DTA_DIR = os.path.join(LEP_ROOTDIR, 'DeepDTA')
    all_data_file = os.path.join(DEEP_DTA_DIR, 'all_structs_info.pkl')

    info_df = pd.read_csv(
        info_file, engine='python',
        usecols=['protein', 'ligand', 'active_struc', 'inactive_struc', 'label', 'SMILES'])

    all_data_df, dataset_splits = process_sharded_datasets(
        info_df, sharded_datasets_dict, all_data_file)

    lengths_df = get_seq_length_stats(all_data_df)

    smiles_df, seqs_df, aff_matrix = process_data_to_DeepDTA(all_data_df)
    # Save to DeepDTA format
    write_DeepDTA_to_files(
        smiles_df, seqs_df, aff_matrix,
        os.path.join(DEEP_DTA_DIR, 'ligands_can.txt'),
        os.path.join(DEEP_DTA_DIR, 'proteins.txt'),
        os.path.join(DEEP_DTA_DIR, 'Y'))

    # Generate test/train splits
    print(f"\nTrain: {len(dataset_splits['train']):}, val: {len(dataset_splits['val']):}, "
          f"test: {len(dataset_splits['test']):}")

    for k, df in dataset_splits.items():
        neg, pos = np.bincount(all_data_df.set_index(['ligand', 'protein']).loc[df]['label'])
        total = neg + pos
        percent = 100.0*pos/total
        print(f"{k:} dataset:\n    Total: {total:}\n    Positive: {pos:} ({percent:.2f}% of total)\n")

    train_folds, test_fold = gen_DeepDTA_train_test_folds(
        smiles_df, seqs_df, aff_matrix, dataset_splits['train'], dataset_splits['test'],
        5,
        os.path.join(DEEP_DTA_DIR, 'folds_scaffold/train_fold_setting1.txt'),
        os.path.join(DEEP_DTA_DIR, 'folds_scaffold/test_fold_setting1.txt'),
        val_split=dataset_splits['val'])
