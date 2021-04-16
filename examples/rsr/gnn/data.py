import numpy as np
import os
import torch
from atom3d.util.transforms import prot_graph_transform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Dataset, DataLoader

    
class GNNTransformRSR(object):
    def __init__(self):
        pass
    def __call__(self, item):
        item = prot_graph_transform(item, ['atoms'], 'scores')
        graph = item['atoms']
        graph.y = torch.FloatTensor([graph.y['score']])
        return graph
    

        
if __name__=="__main__":
    dataset = LMDBDataset('/scratch/users/raphtown/atom3d_mirror/lmdb/RSR/splits/candidates-split-by-time/data/train', transform=GNNTransformRSR())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    # for item in dataset[0]:
    #     print(item, type(dataset[0][item]))
    for item in dataloader:
        print(item)
        break