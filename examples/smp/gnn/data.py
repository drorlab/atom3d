import numpy as np
import os
import torch
from tqdm import tqdm
from atom3d.util.transforms import mol_graph_transform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Dataset, DataLoader

label_mean = {'A': 9.431132871932412, 'B': 1.4056874059499507, 'C': 1.1248823604150906, 'mu': 2.6891224178392688, 'alpha': 75.28570378668626, 
              'homo': -0.24021913527190966, 'lumo': 0.011472716737326845, 'gap': 0.25169191285116993, 'r2': 1190.7002708451198, 'zpve': 0.1489727281138028, 
              'u0': -410.9862782243254, 'u298': -410.97779877528916, 'h298': -410.976854589033, 'g298': -411.019699089552, 'cv': 31.631681642152866, 
              'u0_atom': -2.795832817348651, 'u298_atom': -2.812878342684959, 'h298_atom': -2.8289558033743067, 'g298_atom': -2.601942396274137, 'cv_atom': -22.0908054023777}

label_std = {'A': 1926.3109871136055, 'B': 1.7161127761975044, 'C': 1.132658746068712, 'mu': 1.5010086062061585, 'alpha': 8.16839114317884, 
             'homo': 0.022091350671529012, 'lumo': 0.046816658657652915, 'gap': 0.04724804692482014, 'r2': 279.33132577926256, 'zpve': 0.03317481910943951, 
             'u0': 39.88988551456204, 'u298': 39.88966504402679, 'h298': 39.889665044179694, 'g298': 39.890396233311, 'cv': 4.052262981580719, 
             'u0_atom': 0.3800010697834496, 'u298_atom': 0.3833759718975415, 'h298_atom': 0.38610051482730373, 'g298_atom': 0.3496293149028128, 'cv_atom': 6.10755201901486}
    
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
        return (item['labels'][self.label_mapping[name]] - label_mean[name]) / label_std[name]

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
    data_dir = '/scratch/users/aderry/lmdb/atom3d/small_molecule_properties/splits/split-randomly/data'
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
        print(item.y)
        # torch.save(item, os.path.join(save_dir, 'train', f'data_{i}.pt'))
    
    # for i, item in enumerate(tqdm(val_dataset)):
    #     torch.save(item, os.path.join(save_dir, 'val', f'data_{i}.pt'))
    
    # for i, item in enumerate(tqdm(test_dataset)):
    #     torch.save(item, os.path.join(save_dir, 'test', f'data_{i}.pt'))