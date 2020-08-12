"""Functions for splitting data into test, validation, and training sets."""

import numpy as np

import atom3d.util.log as log

logger = log.get_logger('splits')


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
            train_split (float):
                fraction of data used for training. Default: 0.1
            vali_split (float):
                fraction of data used for validation. Default: 0.1
            test_split (float): fraction of data used for testing. Default: 0.1
            shuffle (bool):     indices are shuffled. Default: True
            random_seed (int):
                specifies random seed for shuffling. Default: None
            exclude (np.array of int):  indices to exclude.


        Returns:
            indices_test (int[]):  indices of the test set.
            indices_vali (int[]):  indices of the validation set.
            indices_train (int[]): indices of the training set.

    """

    # Initialize the indices
    all_indices = np.arange(dataset_size, dtype=int)
    logger.info(f'Splitting dataset with {len(all_indices):} entries.')

    # Delete all indices that shall be excluded
    if exclude is None:
        indices = all_indices
    else:
        logger.info('Excluding', len(exclude), 'entries.')
        to_keep = np.invert(np.isin(all_indices, exclude))
        indices = all_indices[to_keep]
        logger.info('Remaining', len(indices), 'entries.')
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
# split by scaffold
####################################


def scaffold_split(scaffold_list, vali_split=0.1, test_split=0.1):
    """Creates data indices for training and validation splits according to a scaffold split.
        Args:
            scaffold_list (array ofstr): names of the scaffolds
            train_split (float):
                fraction of data used for training. Default: 0.1
            vali_split (float):
                fraction of data used for validation. Default: 0.1
            test_split (float): fraction of data used for testing. Default: 0.1
            random_seed (int):
                specifies random seed for shuffling. Default: None
            exclude (np.array of int):  indices to exclude.
        Returns:
            indices_train (int[]):  indices of the training set.
            indices_vali (int[]):  indices of the validation set.
            indices_test (int[]): indices of the test set.
    """
    
    logger.info(f'Splitting dataset with {len(scaffold_list):} entries.')

    # Calculate the target sizes of the splits
    dataset_size = len(scaffold_list)
    all_indices = np.arange(dataset_size)
    testset_size = test_split * dataset_size
    valiset_size = vali_split * dataset_size
    trainingset_size = dataset_size - valiset_size - testset_size
    
    # Order the scaffolds from common to uncommon 
    scaffolds, counts = np.unique(scaffold_list, return_counts=True)
    order = np.argsort(counts)[::-1]
    scaffolds_ordered = scaffolds[order]
    
    # Initialize index lists
    indices_train = [] 
    indices_vali = [] 
    indices_test = []
    # Initialize counters for scaffolds in each set
    num_sc_train = 0
    num_sc_vali = 0
    num_sc_test = 0

    # Go through the scaffolds from common to uncommon 
    # and fill the training, validation, and test sets
    for sc in scaffolds_ordered:
        # Get all indices of the current scaffold
        scaffold_set = all_indices[np.array(scaffold_list)==sc].tolist()
        # ... and add them to their dataset
        if len(indices_train) < trainingset_size:
            indices_train += scaffold_set
            num_sc_train += 1
        elif len(indices_vali) < valiset_size:
            indices_vali += scaffold_set
            num_sc_vali += 1
        else:
            indices_test += scaffold_set
            num_sc_test += 1
            
    # Report number of scaffolds in each set
    logger.info(f'Scaffolds in the training set: {int(num_sc_train):}')
    logger.info(f'Scaffolds in the validation set: {int(num_sc_vali):}')
    logger.info(f'Scaffolds in the test set: {int(num_sc_test):}')
    
    # Report number of scaffolds in each set
    logger.info(f'Size of the training set: {len(indices_train):}')
    logger.info(f'Size of the validation set: {len(indices_vali):}')
    logger.info(f'Size of the test set: {len(indices_test):}')
    
    return indices_train, indices_vali, indices_test

