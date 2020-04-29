import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import h5py
import sys
sys.path.append('..')
from util import datatypes as dt
from util import file as fi
from util import splits as sp
from get_labels import get_label
from util import graph
from tqdm import tqdm
import os
import torch
from torch_geometric.data import Dataset, Data, DataLoader


def files_exist(files):
    return len(files) != 0 and all([os.path.exists(f) for f in files])

# loader for pytorch-geometric

class GraphPDBBind(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphPDBBind, self).__init__(root, transform, pre_transform)

        self.pdb_idx_dict = self.get_idx_mapping()
    # @property
    # def raw_dir(self):
    #     return self.root

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    # @property
    # def raw_paths(self):
    #     return fi.find_files

    @property
    def processed_file_names(self):
        num_samples = len(self.raw_file_names) // 3
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
    if split_file is not None:
        indices = sp.read_split_file(split_file)

    dataset = GraphPDBBind(root=data_dir)
    # if split specifies pdb ids, convert to indices
    if isinstance(indices[0], str):
        indices = [dataset.pdb_to_idx(x) for x in indices if dataset.pdb_to_idx(x)]
    return DataLoader(dataset.index_select(indices), batch_size)



