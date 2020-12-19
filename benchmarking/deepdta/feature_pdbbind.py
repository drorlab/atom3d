import json
import os
import pathlib
import pickle

import numpy as np
import pandas as pd

from collections import OrderedDict
from Bio.PDB.Polypeptide import PPBuilder
from rdkit import Chem

import atom3d.util.file as fi
import atom3d.util.formats as ft
from atom3d.datasets.lba.process_pdbbind import get_ligand


###############################################################################
### Protein sequence
###############################################################################

def get_seq_from_pdb(pdb_file):
    structure = ft.read_pdb(pdb_file)
    ppb = PPBuilder()
    seq = '\n'.join([str(pp.get_sequence()) for pp in ppb.build_peptides(structure)])
    return (fi.get_pdb_code(pdb_file), seq)


def get_all_seqs_from_pdbs(data_dir, outfile=None):
    all_sequences = {}
    for pdb_file in pathlib.Path(data_dir).rglob('*_protein.pdb'):
        pdb_file = str(pdb_file)
        pdb_code, seq = get_seq_from_pdb(pdb_file)
        all_sequences[pdb_code] = seq
    seqs_df = pd.DataFrame(all_sequences.items(), columns=['pdbcode', 'seq'])
    seqs_df = seqs_df.sort_values(by=['pdbcode'], ascending=[True]).reset_index(drop=True)
    if outfile is not None:
        seqs_df.to_pickle(outfile)
    return seqs_df


def write_seqs_to_fasta(seqs_df, outfile):
    with open(outfile, 'w') as f:
        for _, (pdb_code, seq) in seqs_df.iterrows():
            f.write('>sp|' + pdb_code + '\n')
            f.write(seq + '\n\n\n')


def get_seq_length_stats(seqs_df):
    lengths_df = seqs_df.set_index('pdbcode').seq.apply(len).reset_index(
        name='length')
    print(lengths_df.describe())
    return lengths_df


###############################################################################
### Ligand SMILES
###############################################################################

def convert_ligand_to_smiles(ligand_file):
    ligand = get_ligand(ligand_file)
    return (fi.get_pdb_code(ligand_file), Chem.MolToSmiles(ligand))


def convert_all_ligands_to_smiles(data_dir, outfile=None):
    all_smiles = {}
    for ligand_file in pathlib.Path(data_dir).rglob('*_ligand.sdf'):
        ligand_file = str(ligand_file)
        try:
            pdb_code, smiles = convert_ligand_to_smiles(ligand_file)
            all_smiles[pdb_code] = smiles
        except:
            print('Failed converting to SMILES {:}'.format(ligand_file))
    smiles_df = pd.DataFrame(all_smiles.items(), columns=['pdbcode', 'SMILES'])
    smiles_df = smiles_df.sort_values(by=['pdbcode'],
                                      ascending=[True]).reset_index(drop=True)
    if outfile is not None:
        smiles_df.to_pickle(outfile)
    return smiles_df


###############################################################################
### Binding affinity matrix
###############################################################################
def gen_binding_aff_matrix(smiles_df, seqs_df, labels_df,
                           aff_matrix_outfile=None):
    """
    Binding affinity matrix (drugs x proteins). Each drug-protein pair
    with unknown affinity is indicated as 'nan'.
    """
    labels_df = labels_df.sort_values(by=['pdbcode'],
                                      ascending=[True]).reset_index(drop=True)
    aff_matrix = np.full([len(smiles_df), len(seqs_df)], np.nan)
    for _, (pdb_code, label) in labels_df.iterrows():
        smiles_idx = smiles_df[smiles_df.pdbcode == pdb_code].index[0]
        seq_idx = seqs_df[seqs_df.pdbcode == pdb_code].index[0]
        aff_matrix[smiles_idx][seq_idx] = label
    if aff_matrix_outfile is not None:
        with open(aff_matrix_outfile, 'wb') as f:
            pickle.dump(aff_matrix, f)
    return smiles_df, seqs_df, aff_matrix


###############################################################################
### Helper functions to convert split file into DeepDTA format
###############################################################################

def process_data_to_DeepDTA(data_dir, smiles_file, seqs_file, labels_file,
                            aff_matrix_file):
    # Process ligand SMILES strings
    if (smiles_file is None) or (not os.path.exists(smiles_file)):
        smiles_df = convert_all_ligands_to_smiles(data_dir, smiles_file)
    else:
        smiles_df = pd.read_pickle(smiles_file)
    # Process protein sequences
    if (seqs_file is None) or (not os.path.exists(seqs_file)):
        seqs_df = get_all_seqs_from_pdbs(data_dir, seqs_file)
    else:
        seqs_df = pd.read_pickle(seqs_file)
    # Generate affinity matrix
    if (aff_matrix_file is None) or (not os.path.exists(aff_matrix_file)):
        labels_df = pd.read_csv(labels_file, engine='python', skiprows=1,
                                names=['pdbcode', 'label'])
        smiles_df, seqs_df, aff_matrix = gen_binding_aff_matrix(
            smiles_df, seqs_df, labels_df, aff_matrix_file)
    else:
        with open(aff_matrix_file, 'rb') as f:
            aff_matrix = pickle.load(f)
    return smiles_df, seqs_df, aff_matrix


def write_DeepDTA_to_files(smiles_df, seqs_df, aff_matrix,
                           ligands_file, proteins_file, Y_file):
    with open(ligands_file, 'w') as f:
        ordered_dict = smiles_df.set_index('pdbcode').iloc[:,0].to_dict(into=OrderedDict)
        json.dump(ordered_dict, f)
    with open(proteins_file, 'w') as f:
        ordered_dict = seqs_df.set_index('pdbcode').iloc[:,0].to_dict(into=OrderedDict)
        json.dump(ordered_dict, f)
    with open(Y_file, 'wb') as f:
        pickle.dump(aff_matrix, f)


def read_split_file(split_files):
    if not isinstance(split_files, list):
        split_files = [split_files]
    pdb_codes = []
    for split_file in split_files:
        with open(split_file, 'r') as f:
            pdb_codes.extend([t.strip() for t in f.readlines()])
    return set(pdb_codes)


def read_split_hdf_metadata(metadatas):
    if not isinstance(metadatas, list):
        metadatas = [metadatas]
    pdb_codes = []
    for metadata in metadatas:
        df = pd.read_hdf(metadata)
        pdb_codes.extend(df.ensemble.sort_values().unique())
    return set(pdb_codes)


def gen_DeepDTA_train_test_folds(smiles_df, seqs_df, aff_matrix,
                                 train_pdb_codes, test_pdb_codes,
                                 train_nfold=5, train_fold_file=None,
                                 test_fold_file=None, val_pdb_codes=[]):
    # row: ligand, col: protein
    label_row_inds, label_col_inds = np.where(np.isnan(aff_matrix)==False)
    train_set, val_set, test_set = [], [], []
    for i, (lig_idx, prot_idx) in enumerate(zip(label_row_inds, label_col_inds)):
        pdb_code = seqs_df.iloc[prot_idx].pdbcode
        assert (smiles_df.iloc[lig_idx].pdbcode == pdb_code)
        if pdb_code in train_pdb_codes:
            train_set.append(i)
        elif pdb_code in test_pdb_codes:
            test_set.append(i)
        elif pdb_code in val_pdb_codes:
            val_set.append(i)

    if len(val_pdb_codes) == 0:
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
    PDBBIND_ROOTDIR = '/oak/stanford/groups/rondror/projects/atom3d/supporting_files/ligand_binding_affinity'
    data_dir = os.path.join(PDBBIND_ROOTDIR, 'refined-set')

    DEEP_DTA_DIR = os.path.join(PDBBIND_ROOTDIR, 'DeepDTA')
    smiles_file = os.path.join(DEEP_DTA_DIR, 'pdbbind_v2009_refined_ligand_smiles.pkl')
    seqs_file = os.path.join(DEEP_DTA_DIR, 'pdbbind_v2009_refined_protein_seqs.pkl')
    labels_file = os.path.join(PDBBIND_ROOTDIR, 'initial-dataset/pdbbind_refined_set_labels.csv')
    aff_matrix_file = os.path.join(DEEP_DTA_DIR, 'pdbbind_v2009_refined_aff_matrix.pkl')

    smiles_df, seqs_df, aff_matrix = process_data_to_DeepDTA(
        data_dir, smiles_file, seqs_file, labels_file, aff_matrix_file)

    # Save to DeepDTA format
    write_DeepDTA_to_files(
        smiles_df, seqs_df, aff_matrix,
        os.path.join(DEEP_DTA_DIR, 'ligands_can.txt'),
        os.path.join(DEEP_DTA_DIR, 'proteins.txt'),
        os.path.join(DEEP_DTA_DIR, 'Y'))

    # Generate test/train splits
    #SPLIT_HDF_DIR = '/oak/stanford/groups/rondror/projects/atom3d/ligand_binding_affinity/lba-split-by-sequence-identity-30'
    #SPLIT_HDF_DIR = '/oak/stanford/groups/rondror/projects/atom3d/ligand_binding_affinity/lba-split'
    SPLIT_HDF_DIR = '/oak/stanford/groups/rondror/projects/atom3d/ligand_binding_affinity/lba-split-by-scaffold/'
    split_metadatas = [
        os.path.join(SPLIT_HDF_DIR, 'lba_train_meta_10.h5'),
        os.path.join(SPLIT_HDF_DIR, 'lba_val_meta_10.h5'),
        os.path.join(SPLIT_HDF_DIR, 'lba_test_meta_10.h5'),
    ]
    train_pdb_codes = read_split_hdf_metadata(split_metadatas[0])
    val_pdb_codes = read_split_hdf_metadata(split_metadatas[1])
    test_pdb_codes = read_split_hdf_metadata(split_metadatas[2])
    print(f'Train: {len(train_pdb_codes):}, val: {len(val_pdb_codes):}, test: {len(test_pdb_codes):}')

    train_folds, test_fold = gen_DeepDTA_train_test_folds(
        smiles_df, seqs_df, aff_matrix, train_pdb_codes, test_pdb_codes,
        5,
        os.path.join(DEEP_DTA_DIR, 'folds_scaffold/train_fold_setting1.txt'),
        os.path.join(DEEP_DTA_DIR, 'folds_scaffold/test_fold_setting1.txt'),
        val_pdb_codes=val_pdb_codes)
