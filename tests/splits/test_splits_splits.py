import pytest
import os
import torch
import numpy as np
import atom3d.datasets as da
import atom3d.splits.splits as spl


# -- Test the general split function. --


def test_split():
    # Load LMDB dataset
    dataset = da.load_dataset('tests/test_data/lmdb', 'lmdb')
    # Split with defined indices
    indices_train, indices_val, indices_test = [3,0], [2], [1]
    s = spl.split(dataset, indices_train, indices_val, indices_test)
    train_dataset, val_dataset, test_dataset = s
    # Check whether the frames are in the correct dataset
    assert dataset[0]['atoms'].equals( train_dataset[1]['atoms'] )
    assert dataset[1]['atoms'].equals( test_dataset[0]['atoms'] )
    assert dataset[2]['atoms'].equals( val_dataset[0]['atoms'] )
    assert dataset[3]['atoms'].equals( train_dataset[0]['atoms'] )


# -- Test specific split functions with mock dataset --


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, data, year, scaffold):
        self.data = data
        self.year = year
        self.scaf = scaffold
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        item = {
            'data': self.data[idx],
            'year': self.year[idx],
            'scaffold': self.scaf[idx]
        }
        return item

years = 2020-np.arange(30)
scaffold = [0]*10 + [1]*5 + [2]*3 + [3,4,5]*2 + [6,7,8,9,10,11]
np.random.shuffle(scaffold)
dataset = MockDataset(np.arange(30),years,scaffold)


def test_split_randomly():
    # Perform the split
    s = spl.split_randomly(dataset, random_seed=0)
    train_dataset, val_dataset, test_dataset = s
    # Shuffle with same seed
    np.random.seed(0)
    shuffled = np.arange(30)
    np.random.shuffle(shuffled)
    # Compare split to shuffled data
    assert sum( shuffled[:24] == [i['data'] for i in train_dataset] ) == 24
    assert sum( shuffled[24:27] == [i['data'] for i in val_dataset] ) == 3
    assert sum( shuffled[27:] == [i['data'] for i in test_dataset] ) == 3


def test_split_by_group():
    # Perform the split
    s = spl.split_by_group(dataset, 
                           value_fn=lambda x: x['year'], 
                           train_values=range(1900,2011),
                           val_values=range(2011,2016), 
                           test_values=range(2016,2021))
    train_dataset, val_dataset, test_dataset = s
    # Compare split to what it should be
    assert sum( [i['data'] for i in train_dataset] == np.arange(10,30) ) == 20
    assert sum( [i['data'] for i in val_dataset] == np.arange(5,10) ) == 5
    assert sum( [i['data'] for i in test_dataset] == np.arange(0,5) ) == 5


def test_split_by_group_size():
    # Perform the split
    s = spl.split_by_group_size(dataset, 
                                value_fn=lambda x: x['scaffold'], 
                                val_split=0.2, test_split=0.2)
    train_dataset, val_dataset, test_dataset = s
    # Compare split to what it should be
    assert sum( np.sort([i['scaffold'] for i in train_dataset]) == [0]*10 + [1]*5 + [2]*3 ) == 18
    assert sum( np.sort([i['scaffold'] for i in val_dataset]) == [3,3,4,4,5,5] ) == 6
    assert sum( np.sort([i['scaffold'] for i in test_dataset]) == [6,7,8,9,10,11] ) == 6

