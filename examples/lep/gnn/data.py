import numpy as np
import os
from atom3d.util.transforms import prot_graph_transform, PairedGraphTransform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Dataset, DataLoader
import atom3d.util.graph as gr

    
class GNNTransformLEP(object):
    def __init__(self, atom_keys, label_key):
        self.atom_keys = atom_keys
        self.label_key = label_key
    
    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        item = prot_graph_transform(item, atom_keys=self.atom_keys, label_key=self.label_key)
        
        return item
    

        
if __name__=="__main__":
    dataset = LMDBDataset(os.path.join('/scratch/users/raphtown/atom3d_mirror/lmdb/LEP/splits/split-by-protein/data', 'train'), transform=PairedGraphTransform('atoms_active', 'atoms_inactive', label_key='label'))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for active, inactive in dataloader:
        print(active)
        print(inactive)
        break
    # for item in dataloader:
    #     print(item)
    #     break