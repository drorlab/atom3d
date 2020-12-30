"""Functions for splitting data into test, validation, and training sets."""
from functools import partial

import numpy as np
import torch

import atom3d.util.log as log

logger = log.get_logger('splits')


def split(dataset, indices_train, indices_val, indices_test):
    """Split a dataset into train, validation, and test datasets according to specified indices.

    :param dataset: Dataset to split.
    :type dataset: Dataset
    :param indices_train: List of indices comprising training set.
    :type indices_train: List[int]
    :param indices_val: List of indices comprising validation set.
    :type indices_val: List[int]
    :param indices_test: List of indices comprising test set.
    :type indices_test: List[int]
    :return: Tuple of train, validation, and test datasets
    :rtype: Tuple[Dataset]
    """    
    train_dataset = torch.utils.data.Subset(dataset, indices_train)
    val_dataset = torch.utils.data.Subset(dataset, indices_val)
    test_dataset = torch.utils.data.Subset(dataset, indices_test)

    logger.info(f'Size of the training set: {len(indices_train):}')
    logger.info(f'Size of the validation set: {len(indices_val):}')
    logger.info(f'Size of the test set: {len(indices_test):}')

    return train_dataset, val_dataset, test_dataset


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

def split_randomly(dataset, train_split=None, val_split=0.1, test_split=0.1, random_seed=0):
    """Split a dataset into train, validation and test datasets at random.

    :param dataset: Dataset to split.
    :type dataset: Dataset
    :param train_split: Proportion of data used for training. If None, use all data not in validation or test, defaults to None.
    :type train_split: float, optional
    :param val_split: Proportion of data used for validation, defaults to 0.1
    :type val_split: float, optional
    :param test_split: Proportion of data used for testing, defaults to 0.1
    :type test_split: float, optional
    :param random_seed: Random seed for splitting, defaults to 0
    :type random_seed: int, optional
    :return: Tuple of train, validation, and test datasets
    :rtype: Tuple[Dataset]
    """    

    # Initialize the indices
    num_indices = len(dataset)
    indices = np.arange(num_indices, dtype=int)
    logger.info(f'Splitting dataset with {num_indices:} entries.')

    # Calculate the numbers of elements per split
    num_val = int(np.floor(val_split * num_indices))
    num_test = int(np.floor(test_split * num_indices))
    if train_split is not None:
        num_train = int(np.floor(train_split * num_indices))
    else:
        num_train = num_indices - num_val - num_test

    # Shuffle the dataset indices
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Determine the indices of each split
    indices_train = indices[:num_train]
    indices_val = indices[num_train:(num_train + num_val)]
    indices_test = indices[(num_train + num_val):(num_train + num_val + num_test)]

    return split(dataset, indices_train, indices_val, indices_test)


####################################
# split by group
####################################

def split_by_group(dataset, value_fn, train_values, val_values, test_values):
    """Splits data into train, validation, and test dataset using a value function that maps each data element to a value (or group identifier). These are then used to assign elements to the appropriate splits based on pre-defined lists of values to include in each split.

    :param dataset: Dataset to split.
    :type dataset: Dataset
    :param value_fn: Arbitrary function mapping each data element to a value or group identifier.
    :type value_fn: function
    :param train_values: List of values to include in training set.
    :type train_values: List
    :param val_values: List of values to include in validation set.
    :type val_values: List
    :param test_values: List of values to include in test set.
    :type test_values: List
    :return: Tuple of train, validation, and test datasets
    :rtype: Tuple[Dataset]
    """    

    values = [value_fn(x) for x in dataset]

    # Determine the indices of each split
    indices_train = [i for i,x in enumerate(values) if x in train_values]
    indices_val = [i for i,x in enumerate(values) if x in val_values]
    indices_test = [i for i,x in enumerate(values) if x in test_values]
    return split(dataset, indices_train, indices_val, indices_test)


####################################
# split by group size
####################################


def split_by_group_size(dataset, value_fn, val_split=0.1, test_split=0.1):
    """Splits data into train, validation, and test dataset using a value function that maps each data element to a value (or group identifier). Elements are grouped by the value returned by this value function, and groups are sorted by size. 
    The groups are then added first to train, then to validation, then to test splits in order of group size, so that the largest groups (i.e. most common examples)  are in train and the smallest groups (i.e. less common examples) are in test. 
    Each split is filled iteratively up to the sizes specified by ``val_split`` and ``test_split``.

    :param dataset: Dataset to split.
    :type dataset: Dataset
    :param value_fn: Arbitrary function mapping each data element to a value or group identifier.
    :type value_fn: function
    :param val_split: Proportion of data used for validation, defaults to 0.1
    :type val_split: float, optional
    :param test_split: Proportion of data used for testing, defaults to 0.1
    :type test_split: float, optional
    :return: Tuple of train, validation, and test datasets.
    :rtype: Tuple[Dataset]
    """    

    values = [value_fn(x) for x in dataset]

    logger.info(f'Splitting dataset with {len(values):} entries.')

    # Calculate the target sizes of the splits
    dataset_size = len(values)
    all_indices = np.arange(dataset_size)
    test_size = test_split * dataset_size
    val_size = val_split * dataset_size
    train_size = dataset_size - val_size - test_size

    # Order the scaffolds from common to uncommon 
    unique_values, counts = np.unique(values, return_counts=True)
    order = np.argsort(counts)[::-1]
    values_ordered = unique_values[order]

    # Initialize index lists
    indices_train = []
    indices_val = []
    indices_test = []
    # Initialize counters for scaffolds in each set
    num_sc_train = 0
    num_sc_val = 0
    num_sc_test = 0

    # Go through the scaffolds from common to uncommon 
    # and fill the training, validation, and test sets
    for sc in values_ordered:
        # Get all indices of the current scaffold
        curr_group = all_indices[np.array(values) == sc].tolist()
        # ... and add them to their dataset
        if len(indices_train) < train_size:
            indices_train += curr_group
            num_sc_train += 1
        elif len(indices_val) < val_size:
            indices_val += curr_group
            num_sc_val += 1
        else:
            indices_test += curr_group
            num_sc_test += 1

    # Report number of scaffolds in each set
    logger.info(f'Groups in the training set: {int(num_sc_train):}')
    logger.info(f'Groups in the validation set: {int(num_sc_val):}')
    logger.info(f'Groups in the test set: {int(num_sc_test):}')

    # Report number of scaffolds in each set
    return split(dataset, indices_train, indices_val, indices_test)

    
####################################
# frequently used specific splits
####################################

split_by_year = partial(split_by_group, value_fn=lambda x: x['year'])

split_by_scaffold = partial(split_by_group_size, value_fn=lambda x: x['scaffold'])
