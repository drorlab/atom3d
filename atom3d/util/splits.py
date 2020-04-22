# Splits data into test, validation, and training sets.
import numpy as np

def random_split(dataset_size,train_split=None,vali_split=0.1,test_split=0.1,shuffle=True,random_seed=None):
    """Creates data indices for training and validation splits.
        
        Args:
            dataset_size (int): number of elements in the dataset
            vali_split (float): fraction of data used for validation. Default: 0.1
            test_split (float): fraction of data used for testing. Default: 0.1
            shuffle (bool):     indices are shuffled. Default: True
            random_seed (int):  specifies random seed for shuffling. Default: None
            
        Returns:
            indices_test (int[]):  indices of the test set.
            indices_vali (int[]):  indices of the validation set.
            indices_train (int[]): indices of the training set.
            
    """
        
    indices = np.arange(dataset_size,dtype=int)
        
    # Calculate the numbers of elements per split
    vsplit = int(np.floor(vali_split * dataset_size))
    tsplit = int(np.floor(test_split * dataset_size))
    if train_split is not None:
        train = int(np.floor(train_split * dataset_size))
    else:
        train = dataset_size-vsplit-tsplit
        
    # Shuffle the dataset if desired
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Determine the indices of each split
    indices_test  = indices[:tsplit]
    indices_vali  = indices[tsplit:tsplit+vsplit]
    indices_train = indices[tsplit+vsplit:tsplit+vsplit+train]
        
    return indices_test, indices_vali, indices_train

