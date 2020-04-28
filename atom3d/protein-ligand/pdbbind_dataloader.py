import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import h5py
import atom3d.util.datatypes as dt


class PepBind(data.Dataset):
    def __init__(self, split, split_method, hdf_file):
        self.proteins = pd.read_hdf(hdf_file, 'proteins')
        self.pockets = pd.read_hdf(hdf_file, 'pockets')
        self.pdb_codes = pd.read_hdf(hdf_file, 'pdb_codes')
        self.ligands = pd.read_hdf(hdf_file, 'ligands')['Mol']
            
        # TODO: split dataset
        
        # TODO: featurize proteins and ligands for GNN
            
    def __getitem__(self, i):
        pdb_code = self.pdb_codes.iloc[i]
        pocket = self.pockets.iloc[i]
        ligand = self.ligand.iloc[i]
        
        return pdb_code, pocket, ligand
    
    def __len__(self):
        return len(self.pdb_codes)

    
def pdbbind_dataloader(split, hdf_file, batch_size, shuffle, num_workers):
    return data.DataLoader(dataset=PepBind(split, split_method, hdf_file), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)