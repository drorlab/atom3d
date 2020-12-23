"""TODO: This code has been significantly re-written and should be tested."""
import math
import random

import numpy as np

import atom3d.protein.sequence as seq
import atom3d.util.file as fi
import atom3d.util.log as log
import atom3d.splits.splits as splits

logger = log.get_logger('sequence_splits')

####################################
# split by pre-clustered sequence
# identity clusters from PDB
####################################


def cluster_split(dataset, cutoff, val_split=0.1,
                  test_split=0.1, min_fam_in_split=5, random_seed=None):
    """
    Splits pdb dataset using pre-computed sequence identity clusters from PDB.

    Generates train, val, test sets.

    We assume there is one PDB code per entry in dataset.

    Args:
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

    logger.info('Loading chain sequences')
    all_chain_sequences = [seq.get_chain_sequences(x['atoms']) for x in dataset]

    pdb_codes = np.array([fi.get_pdb_code(x[0][0][0]) for x in all_chain_sequences])
    n_orig = len(np.unique(pdb_codes))
    clusterings = seq.get_pdb_clusters(cutoff, np.unique(pdb_codes))

    # If code not present in clustering, we don't use.
    to_use = [i for (i, x) in enumerate(pdb_codes) if x in clusterings[0]]
    n = len(np.unique(pdb_codes[to_use]))
    to_use = set(to_use)

    logger.info(
        f'Removing {n_orig - n:} / {n_orig:} '
        f'sequences due to not finding in clustering.')

    test_size = n * test_split
    val_size = n * val_split

    logger.info('generating validation set...')
    val_indices, to_use = _create_cluster_split(
        all_chain_sequences, clusterings, to_use, val_size, min_fam_in_split)
    logger.info('generating test set...')
    test_indices, to_use = _create_cluster_split(
        all_chain_sequences, clusterings, to_use, test_size, min_fam_in_split)
    train_indices = to_use

    return splits.split(dataset, train_indices, val_indices, test_indices)


def _create_cluster_split(all_chain_sequences, clusterings, to_use, split_size, min_fam_in_split):
    """
    Create a split while retaining diversity specified by min_fam_in_split.
    Returns split and removes any pdbs in this split from the remaining dataset
    """
    dataset_size = len(all_chain_sequences)
    code_to_idx = {fi.get_pdb_code(y[0][0]): i for (i, x) in enumerate(all_chain_sequences) for y in x}

    all_indices = set(range(dataset_size))
    split, used = set(), all_indices.difference(to_use)
    while len(split) < split_size:
        i = random.sample(to_use, 1)[0]
        pdb_code = fi.get_pdb_code(all_chain_sequences[i][0][0][0])
        found = seq.find_cluster_members(pdb_code, clusterings)

        # Map back to source.
        found = set([code_to_idx[x] for x in found])
        found = found.difference(used)

        # ensure that at least min_fam_in_split families in each split
        max_fam_size = int(math.ceil(split_size / min_fam_in_split))
        split = split.union(list(found)[:max_fam_size])
        to_use = to_use.difference(found)
        used = used.union(found)

    return split, to_use


####################################
# split by calculating sequence identity
# to any example in training set
####################################


def identity_split(
        dataset, cutoff, val_split=0.1, test_split=0.1,
        min_fam_in_split=5, blast_db=None, random_seed=None):
    """
    Splits pdb dataset using pre-computed sequence identity clusters from PDB.

    Generates train, val, test sets.

    Args:
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
    all_chain_sequences = [seq.get_chain_sequences(x['atoms']) for x in dataset]
    # Flatten.
    flat_chain_sequences = [x for sublist in all_chain_sequences for x in sublist]

    if blast_db is None:
        seq.write_to_blast_db(flat_chain_sequences, 'blast_db')
        blast_db = 'blast_db'

    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(dataset)
    test_size = n * test_split
    val_size = n * val_split

    to_use = set(range(len(all_chain_sequences)))
    logger.info('generating validation set...')
    val_indices, to_use = _create_identity_split(
        all_chain_sequences, cutoff, to_use, val_size, min_fam_in_split, blast_db)
    logger.info('generating test set...')
    test_indices, to_use = _create_identity_split(
        all_chain_sequences, cutoff, to_use, test_size, min_fam_in_split, blast_db)
    train_indices = to_use

    return splits.split(dataset, train_indices, val_indices, test_indices)


def _create_identity_split(all_chain_sequences, cutoff, to_use, split_size,
                           min_fam_in_split, blast_db):
    """
    Create a split while retaining diversity specified by min_fam_in_split.
    Returns split and removes any pdbs in this split from the remaining dataset
    """
    dataset_size = len(all_chain_sequences)
    chain_to_idx = {y[0]: i for (i, x) in enumerate(all_chain_sequences) for y in x}

    all_indices = set(range(dataset_size))
    split, used = set(), all_indices.difference(to_use)
    while len(split) < split_size:
        i = random.sample(to_use, 1)[0]

        # Get chains that match.
        found = seq.find_similar(all_chain_sequences[i], blast_db, cutoff, dataset_size)
        # Map back to source.
        found = set([chain_to_idx[x] for x in found])
        found = found.difference(used)

        # ensure that at least min_fam_in_split families in each split
        max_fam_size = int(math.ceil(split_size / min_fam_in_split))
        split = split.union(list(found)[:max_fam_size])
        to_use = to_use.difference(found)
        used = used.union(found)

    return split, to_use