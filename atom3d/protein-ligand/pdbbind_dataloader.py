import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from util import datatypes as dt
from util import file as fi
from util import splits as sp
from get_labels import get_label
from util import graph
import os
import torch
from torch_geometric.data import Dataset, Data, DataLoader


# loader for pytorch-geometric

class GraphPDBBind(Dataset):
    """
    PDBBind dataset in pytorch-geometric format. 
    Ref: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphPDBBind, self).__init__(root, transform, pre_transform)

        self.pdb_idx_dict = self.get_idx_mapping()

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        num_samples = len(self.raw_file_names) // 3 # each example has protein/pocket/ligand files
        return [f'data_{i}.pt' for i in range(num_samples)]

    def get_idx_mapping(self):
        pdb_idx_dict = {}
        i = 0
        for file in self.raw_file_names:
            if '_pocket' in file:
                pdb_code = fi.get_pdb_code(file)
                pdb_idx_dict[pdb_code] = i
                i += 1
        return pdb_idx_dict


    def pdb_to_idx(self, pdb):
        return self.pdb_idx_dict.get(pdb)

    def process(self):
        label_file = os.path.join(self.root, 'pdbbind_refined_set_labels.csv')
        label_df = pd.read_csv(label_file)
        i = 0
        for raw_path in self.raw_paths:
            if '_pocket' in raw_path:
                pdb_code = fi.get_pdb_code(raw_path)
                x, edge_index, pos = graph.prot_df_to_graph(dt.bp_to_df(dt.read_any(raw_path, name=pdb_code)))
                y = get_label(pdb_code, label_df)
                data = Data(x, edge_index, pos, y)
                torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1
            else:
                continue

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


def pdbbind_dataloader(batch_size, data_dir='../data/PDBBind', split_file=None):
    """
    Creates dataloader for PDBBind dataset with specified split. 
    Assumes pre-computed split in 'split_file', which is used to index Dataset object
    TODO: implement on-the-fly splitting using split functions
    """
    if split_file is not None:
        indices = sp.read_split_file(split_file)

    dataset = GraphPDBBind(root=data_dir)
    # if split specifies pdb ids, convert to indices
    if isinstance(indices[0], str):
        indices = [dataset.pdb_to_idx(x) for x in indices if dataset.pdb_to_idx(x)]
    return DataLoader(dataset.index_select(indices), batch_size)



