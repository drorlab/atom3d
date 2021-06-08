import numpy as np
import os
import torch
from tqdm import tqdm
from atom3d.util.transforms import mol_graph_transform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Dataset, DataLoader

    
class GNNTransformSMP(object):
    def __init__(self, label_name):
        self.label_name = label_name
    def _lookup_label(self, item, name):
        if 'label_mapping' not in self.__dict__:
            label_mapping = [
                'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
                'u0', 'u298', 'h298', 'g298', 'cv',
                'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom', 'cv_atom',
                ]
            self.label_mapping = {k: v for v, k in enumerate(label_mapping)}
        return item['labels'][self.label_mapping[name]]

    def __call__(self, item):
        item = mol_graph_transform(item, 'atoms', 'labels', allowable_atoms=['C', 'H', 'O', 'N', 'F'], use_bonds=True, onehot_edges=True)
        graph = item['atoms']
        x2 = torch.tensor(item['atom_feats'], dtype=torch.float).t().contiguous()
        graph.x = torch.cat([graph.x.to(torch.float), x2], dim=-1)
        graph.y = self._lookup_label(item, self.label_name)
        graph.id = item['id']
        return graph
    

        
if __name__=="__main__":
    save_dir = '/scratch/users/aderry/atom3d/smp'
    data_dir = '/scratch/users/aderry/lmdb/atom3d/small_molecule_properties'
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    train_dataset = LMDBDataset(os.path.join(data_dir, 'train'), transform=GNNTransformSMP(label_name='mu'))
    # val_dataset = LMDBDataset(os.path.join(data_dir, 'val'), transform=GNNTransformSMP())
    # test_dataset = LMDBDataset(os.path.join(data_dir, 'test'), transform=GNNTransformSMP())
    
    # train_loader = DataLoader(train_dataset, 1, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4)
    # for item in dataset[0]:
    #     print(item, type(dataset[0][item]))
    for i, item in enumerate(tqdm(train_dataset)):
        break
        # torch.save(item, os.path.join(save_dir, 'train', f'data_{i}.pt'))
    
    # for i, item in enumerate(tqdm(val_dataset)):
    #     torch.save(item, os.path.join(save_dir, 'val', f'data_{i}.pt'))
    
    # for i, item in enumerate(tqdm(test_dataset)):
    #     torch.save(item, os.path.join(save_dir, 'test', f'data_{i}.pt'))