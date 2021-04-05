import numpy as np
import os
from atom3d.util.transforms import prot_graph_transform, mol_graph_transform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Dataset, DataLoader
import atom3d.util.graph as gr

    
class GNNTransformLBA(object):
    def __init__(self, pocket_only=True):
        self.pocket_only = pocket_only
    
    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        if self.pocket_only:
            item = prot_graph_transform(item, atom_keys=['atoms_pocket'], label_key='scores')
        else:
            item = prot_graph_transform(item, atom_keys=['atoms_protein', 'atoms_pocket'], label_key='scores')
        # transform ligand into PTG graph
        item = mol_graph_transform(item, 'atoms_ligand', 'scores', use_bonds=True)
        node_feats, edges, edge_feats, node_pos = gr.combine_graphs(item['atoms_pocket'], item['atoms_ligand'])
        combined_graph = Data(node_feats, edges, edge_feats, y=item['scores']['neglog_aff'], pos=node_pos)
        return combined_graph
    

        
if __name__=="__main__":
    dataset = LMDBDataset('/scratch/users/aderry/lmdb/atom3d/lba_lmdb/splits/split-by-sequence-identity-30/data/train', transform=GNNTransformLBA())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    # for item in dataset[0]:
    #     print(item, type(dataset[0][item]))
    for item in dataloader:
        print(item)
        break