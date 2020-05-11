"""Functions for splitting data into test, validation, and training sets."""
import sys

import numpy as np

import atom3d.util.file as fi
import atom3d.util.sequence as seq

sys.path.append('../..')


def read_split_file(split_file):
    """
    Read text file with pre-defined split, returning list of examples.

    One example per row in text file.
    """
    with open(split_file) as f:
        # file may contain integer indices or string identifiers (e.g. PDB
        # codes)
        lines = f.readlines()
        try:
            split = [int(x.strip()) for x in lines]
        except ValueError:
            split = [x.strip() for x in lines]
    return split


####################################
# split randomly
####################################


def random_split(dataset_size, train_split=None, vali_split=0.1,
                 test_split=0.1, shuffle=True, random_seed=None, exclude=None):
    """Creates data indices for training and validation splits.

        Args:
            dataset_size (int): number of elements in the dataset
            vali_split (float):
                fraction of data used for validation. Default: 0.1
            test_split (float): fraction of data used for testing. Default: 0.1
            shuffle (bool):     indices are shuffled. Default: True
            random_seed (int):
                specifies random seed for shuffling. Default: None

        Returns:
            indices_test (int[]):  indices of the test set.
            indices_vali (int[]):  indices of the validation set.
            indices_train (int[]): indices of the training set.

    """

    # Initialize the indices
    all_indices = np.arange(dataset_size, dtype=int)
    print('Splitting dataset with', len(all_indices), 'entries.')

    # Delete all indices that shall be excluded
    if exclude is None:
        indices = all_indices
    else:
        print('Excluding', len(exclude), 'entries.')
        to_keep = np.invert(np.isin(all_indices, exclude))
        indices = all_indices[to_keep]
        print('Remaining', len(indices), 'entries.')
    num_indices = len(indices)

    # Calculate the numbers of elements per split
    vsplit = int(np.floor(vali_split * num_indices))
    tsplit = int(np.floor(test_split * num_indices))
    if train_split is not None:
        train = int(np.floor(train_split * num_indices))
    else:
        train = num_indices - vsplit - tsplit

    # Shuffle the dataset if desired
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Determine the indices of each split
    indices_test = indices[:tsplit]
    indices_vali = indices[tsplit:tsplit + vsplit]
    indices_train = indices[tsplit + vsplit:tsplit + vsplit + train]

    return indices_test, indices_vali, indices_train


####################################
# split by time
####################################


def time_split(data, val_years, test_years):
    """
    Splits data into train, val, test by year.

    Args:
        data (DataFrame): year data, with columns named 'pdb' and 'year'
        val_years (str[]): years to include in validation set
        test_years (str[]): years to include in test set

    Returns:
        train_set (str[]):  pdbs in the train set
        val_set (str[]):  pdbs in the validation set
        test_set (str[]): pdbs in the test set
    """
    val = data[data.year.isin(val_years)]
    test = data[data.year.isin(test_years)]
    train = data[~data.pdb.isin(val.pdb.tolist() + test.pdb.tolist())]

    train_set = train['pdb'].tolist()
    val_set = val['pdb'].tolist()
    test_set = test['pdb'].tolist()

    return train_set, val_set, test_set


####################################
# split by pre-clustered sequence
# identity clusters from PDB
####################################

def cluster_split(all_chain_sequences, cutoff, val_split=0.1,
                  test_split=0.1, min_fam_in_split=5, random_seed=None):
    """
    Splits pdb dataset using pre-computed sequence identity clusters from PDB.

    Generates train, val, test sets.

    Args:
        all_chain_sequences ((str, chain_sequences)[]):
            tuple of pdb ids and chain_sequences in dataset
        cutoff (float):
            sequence identity cutoff (can be .3, .4, .5, .7, .9, .95, 1.0)
        val_split (float): fraction of data used for validation. Default: 0.1
        test_split (float): fraction of data used for testing. Default: 0.1
        min_fam_in_split (int): controls variety of val/test sets. Default: 5
        random_seed (int):  specifies random seed for shuffling. Default: None

    Returns:
        train_set (str[]):  pdbs in the train set
        val_set (str[]):  pdbs in the validation set
        test_set (str[]): pdbs in the test set

    """
    if random_seed is not None:
        np.random.seed(random_seed)

    pdb_codes = \
        np.unique([fi.get_pdb_code(x[0]) for (x, _) in all_chain_sequences])
    n = len(pdb_codes)
    clusterings = seq.get_pdb_clusters(cutoff, pdb_codes)
    test_size = n * test_split
    val_size = n * val_split
    max_hit_size_test = test_size / min_fam_in_split
    max_hit_size_val = val_size / min_fam_in_split

    np.random.shuffle(all_chain_sequences)

    print('generating validation set...')
    val_set, all_chain_sequences = create_cluster_split(
        all_chain_sequences, clusterings, cutoff, val_size, min_fam_in_split)
    print('generating test set...')
    test_set, all_chain_sequences = create_cluster_split(
        all_chain_sequences, clusterings, cutoff, test_size, min_fam_in_split)
    train_set = all_chain_sequences

    train_set = [x[0] for x in train_set]
    val_set = [x[0] for x in val_set]
    test_set = [x[0] for x in test_set]

    print('train size', len(train_set))
    print('val size', len(val_set))
    print('test size', len(test_set))

    return train_set, val_set, test_set


def create_cluster_split(all_chain_sequences, clusterings, cutoff, split_size,
                         min_fam_in_split):
    """
    Create a split while retaining diversity specified by min_fam_in_split.
    Returns split and removes any pdbs in this split from the remaining dataset
    """
    pdb_ids = np.array(
        [fi.get_pdb_code(p[0]) for (p, _) in all_chain_sequences])
    split = set()
    idx = 0
    while len(split) < split_size:
        (rand_id, _) = all_chain_sequences[idx]
        pdb_code = fi.get_pdb_code(rand_id[0])
        split.add(pdb_code)
        hits = seq.find_cluster_members(pdb_code, clusterings)
        # ensure that at least min_fam_in_split families in each split
        if len(hits) > split_size / min_fam_in_split:
            idx += 1
            continue
        split = split.union(hits)
        idx += 1

    matches = np.array([i for i, x in enumerate(pdb_ids) if x in split])
    selected_chain_sequences = \
        [x for i, x in enumerate(all_chain_sequences) if i in matches]
    remaining_chain_sequences = \
        [x for i, x in enumerate(all_chain_sequences) if i not in matches]

    return selected_chain_sequences, remaining_chain_sequences


####################################
# split by calculating sequence identity
# to any example in training set
####################################

def identity_split(
        all_chain_sequences, cutoff, val_split=0.1, test_split=0.1,
        min_fam_in_split=5, blast_db=None, random_seed=None):
    """
    Splits pdb dataset using pre-computed sequence identity clusters from PDB.

    Generates train, val, test sets.

    Args:
        all_chain_sequences ((str, chain_sequences)[]):
            tuple of pdb ids and chain_sequences in dataset
        cutoff (float):
            sequence identity cutoff (can be .3, .4, .5, .7, .9, .95, 1.0)
        val_split (float): fraction of data used for validation. Default: 0.1
        test_split (float): fraction of data used for testing. Default: 0.1
        min_fam_in_split (int): controls variety of val/test sets. Default: 5
        blast_db (str):
            location of pre-computed BLAST DB for dataset. If None, compute and
            save in 'blast_db'. Default: None
        random_seed (int):  specifies random seed for shuffling. Default: None

    Returns:
        train_set (str[]):  pdbs in the train set
        val_set (str[]):  pdbs in the validation set
        test_set (str[]): pdbs in the test set

    """
    if blast_db is None:
        seq.write_to_blast_db(all_chain_sequences, 'blast_db')
        blast_db = 'blast_db'

    if random_seed is not None:
        np.random.seed(random_seed)

    pdb_codes = \
        np.unique([fi.get_pdb_code(x[0]) for (x, _) in all_chain_sequences])
    n = len(pdb_codes)
    test_size = n * test_split
    val_size = n * val_split
    max_hit_size_test = test_size / min_fam_in_split
    max_hit_size_val = val_size / min_fam_in_split

    np.random.shuffle(all_chain_sequences)

    print('generating validation set...')
    val_set, all_chain_sequences = create_identity_split(
        all_chain_sequences, cutoff, val_size, min_fam_in_split)
    print('generating test set...')
    test_set, all_chain_sequences = create_identity_split(
        all_chain_sequences, cutoff, test_size, min_fam_in_split)
    train_set = all_chain_sequences

    train_set = [x[0] for x in train_set]
    val_set = [x[0] for x in val_set]
    test_set = [x[0] for x in test_set]

    print('train size', len(train_set))
    print('val size', len(val_set))
    print('test size', len(test_set))

    return train_set, val_set, test_set


def create_identity_split(all_chain_sequences, cutoff, split_size,
                          min_fam_in_split):
    """
    Create a split while retaining diversity specified by min_fam_in_split.
    Returns split and removes any pdbs in this split from the remaining dataset
    """
    dataset_size = len(all_chain_sequences)
    pdb_ids = np.array(
        [fi.get_pdb_code(p[0]) for (p, _) in all_chain_sequences])
    split = set()
    idx = 0
    while len(split) < split_size:
        (rand_id, rand_cs) = all_chain_sequences[idx]
        split.add(rand_id)
        hits = seq.find_similar(rand_cs, 'blast_db', cutoff, dataset_size)
        # ensure that at least min_fam_in_split families in each split
        if len(hits) > split_size / min_fam_in_split:
            idx += 1
            continue
        split = split.union(hits)
        idx += 1

    matches = np.array([i for i, x in enumerate(pdb_ids) if x in split])
    selected_chain_sequences = \
        [x for i, x in enumerate(all_chain_sequences) if i in matches]
    remaining_chain_sequences = \
        [x for i, x in enumerate(all_chain_sequences) if i not in matches]

    return selected_chain_sequences, remaining_chain_sequences
