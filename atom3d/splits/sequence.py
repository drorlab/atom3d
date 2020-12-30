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
    """Splits pdb dataset using pre-computed sequence identity clusters from PDB, ensuring that no cluster spans multiple splits. 

    Clusters are selected randomly into validation and test sets, but to ensure that there is some diversity in each set (i.e. a split does not consist of a single sequence cluster), a minimum number of clusters in each split is enforced. . Some data examples may be removed in order to satisfy this constraint.
    
    This function assumes that the PDB code or PDB filename exists in the ``ensemble`` field of the ``atoms`` dataframe in the dataset.

    :param dataset: Dataset to perform the split on.
    :type dataset: ATOM3D Dataset
    :param cutoff: Sequence identity cutoff. Possible values: 0.3, 0.4, 0.5, 0.7, 0.9, 0.95, 1.0
    :type cutoff: float
    :param val_split: Fraction of data used in validation set, defaults to 0.1
    :type val_split: float, optional
    :param test_split: Fraction of data used in test set, defaults to 0.1
    :type test_split: float, optional
    :param min_fam_in_split: Minimum number of sequence clusters to be included in validation and test sets, defaults to 5
    :type min_fam_in_split: int, optional
    :param random_seed: Random seed for sampling clusters, defaults to None
    :type random_seed: int, optional

    :return: Tuple containing training, validation, and test sets, each as ATOM3D Dataset objects.
    :rtype: Tuple[Dataset]
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
    Helper function for :func:`create_cluster_split`. Creates a single split of ``split_size`` elements while retaining diversity specified by ``min_fam_in_split``.
    Takes in ``all_chain_sequences`` and reference ``clusterings`` from PDB, as well as a list of valid indices to sample from, specified by ``to_use``.
    Returns indices of new split and indices remaining in dataset after removing those used in split.
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
    """Splits a dataset of proteins by sequence identity at specified cutoff value. Proteins are randomly selected to be placed in validation and test splits, along with all proteins within ``cutoff`` sequence identity (calculated by BLAST).

        To ensure that there is some diversity in each set (i.e. a split does not consist of a single sequence cluster), a minimum number of clusters in each split is enforced. Some data examples may be removed in order to satisfy this constraint.

        Note that the construction of this function means that it is effectively a cluster split. All examples within ``cutoff`` of the sampled query protein are added to validation set, meaning that some examples near the edge of the cluster may in fact share less than ``cutoff`` sequence identity with other proteins in the dataset.
        Therefore, this does not satisfy the constraints for a strict sequence identity cutoff: that 
        (1) no protein in validation split shares greater than ``cutoff`` sequence identity with any protein in the train set, and 
        (2) no protein in the test split shares greater than ``cutoff`` sequence identity with any protein in either train or validation sets.
        A function that satisfies this more strict definition is currently under development.



    :param dataset: Dataset to perform the split on.
    :type dataset: ATOM3D Dataset
    :param cutoff: Sequence identity cutoff, between 0 and 1
    :type cutoff: float
    :param val_split: Fraction of data used in validation set, defaults to 0.1
    :type val_split: float, optional
    :param test_split: Fraction of data used in test set, defaults to 0.1
    :type test_split: float, optional
    :param min_fam_in_split: Minimum number of sequence clusters to be included in validation and test sets, defaults to 5
    :type min_fam_in_split: int, optional
    :param random_seed: Random seed for sampling clusters, defaults to None
    :type random_seed: int, optional

    :return: Tuple containing training, validation, and test sets, each as ATOM3D Dataset objects.
    :rtype: Tuple[Dataset]
    """        

    all_chain_sequences = [seq.get_chain_sequences(x['atoms']) for x in dataset]
    # Flatten.
    flat_chain_sequences = [x for sublist in all_chain_sequences for x in sublist]

    # write all sequences to BLAST-formatted database
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
    Helper function for :func:`create_cluster_split`. Creates a single split of ``split_size`` elements while retaining diversity specified by ``min_fam_in_split``.
    Takes in ``all_chain_sequences`` and reference ``blast_db``, as well as a list of valid indices to sample from, specified by ``to_use``.
    Returns indices of new split and indices remaining in dataset after removing those used in split.
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