import numpy as np
import os
import torch
from tqdm import tqdm
from atom3d.util.transforms import prot_graph_transform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Dataset, DataLoader

    
class GNNTransformPSR(object):
    def __init__(self):
        pass
    def __call__(self, item):
        item = prot_graph_transform(item, ['atoms'], 'scores')
        graph = item['atoms']
        graph.y = torch.FloatTensor([graph.y['gdt_ts']])
        split = eval(item['id'])
        graph.target = split[0]
        graph.decoy = split[1]
        return graph
    

        
if __name__=="__main__":
    save_dir = '/scratch/users/aderry/atom3d/psr'
    data_dir = '/scratch/users/raphtown/atom3d_mirror/lmdb/PSR/splits/split-by-year/data'
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    train_dataset = LMDBDataset(os.path.join(data_dir, 'train'), transform=GNNTransformPSR())
    val_dataset = LMDBDataset(os.path.join(data_dir, 'val'), transform=GNNTransformPSR())
    test_dataset = LMDBDataset(os.path.join(data_dir, 'test'), transform=GNNTransformPSR())
    
    train_loader = DataLoader(train_dataset, 1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4)

    
    print('processing train dataset...')
    for i, item in enumerate(tqdm(train_dataset)):
        torch.save(item, os.path.join(save_dir, 'train', f'data_{i}.pt'))
    
    print('processing validation dataset...')
    for i, item in enumerate(tqdm(val_dataset)):
        torch.save(item, os.path.join(save_dir, 'val', f'data_{i}.pt'))
    
    print('processing test dataset...')
    for i, item in enumerate(tqdm(test_dataset)):
        torch.save(item, os.path.join(save_dir, 'test', f'data_{i}.pt'))