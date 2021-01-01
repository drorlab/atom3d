import torch
import numpy as np
import atom3d.datasets as da
from pathlib import Path

class MockDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = np.arange(30)
        self.year = 2020-np.arange(30)
        self.scaf = [0]*10 + [1]*5 + [2]*3 + [3,4,5]*2 + [6,7,8,9,10,11]
        np.random.shuffle(self.scaf)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        item = {
            'data': self.data[idx],
            'year': self.year[idx],
            'scaffold': self.scaf[idx]
        }
        return item

def load_example_dataset():
    dataset = da.load_dataset(str(Path(__file__).parent.absolute()) + '/test_lmdb', 'lmdb')
    return dataset