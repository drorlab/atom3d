import os
import logging
import numpy as np
import scipy as sp
import scipy.spatial
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from atom3d.datasets import LMDBDataset, extract_coordinates_as_numpy_arrays
from utils import batch_stack, drop_zeros


class CormorantDatasetLEP(Dataset):
    """
    Data structure for a Cormorant dataset. Extends PyTorch Dataset.

    :param data: Dictionary of arrays containing molecular properties.
    :type data: dict
    :param shuffle: If true, shuffle the points in the dataset.
    :type shuffle: bool, optional
        
    """
    def __init__(self, data, included_species=None, shuffle=False):
        # Define data
        self.data = data
        # Get the size of all parts of the dataset
        ds_sizes = [len(self.data[key]) for key in self.data.keys()]
        # Make sure all parts of the dataset have the same length
        for size in ds_sizes[1:]: assert size == ds_sizes[0]
        # Set the dataset size
        self.num_pts = ds_sizes[0]
        # If included species is not specified
        if included_species is None:
            all_charges = np.concatenate(self.data['charges_active'], self.data['charges_inactive'])
            self.included_species = torch.unique(all_charges, sorted=True)
        else:
            self.included_species = torch.unique(included_species, sorted=True)
        # Convert charges to one-hot representation
        self.data['one_hot_active'] = self.data['charges_active'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        self.data['one_hot_inactive'] = self.data['charges_inactive'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        # Calculate parameters
        self.num_species = len(included_species)
        self.max_charge = max(included_species)
        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}
        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()
        if shuffle:
            self.perm = torch.randperm(self.num_pts)
        else:
            self.perm = None

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}
        print(self.stats)

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]
        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}


def collate_lep(batch):
    """
    Collates LEP datapoints into the batch format for Cormorant.
    
    :param batch: The data to be collated.
    :type batch: list of datapoints

    :param batch: The collated data.
    :type batch: dict of Pytorch tensors

    """
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}
    # Define which fields to keep 
    to_keep1 = (batch['charges_active'].sum(0) > 0)
    to_keep2 = (batch['charges_inactive'].sum(0) > 0)
    # Start building the new batch
    new_batch = {}
    # Copy label data. 
    new_batch['label'] = batch['label']
    # Split structural data and drop zeros
    for key in ['charges','positions','one_hot']:
        new_batch[key+'1'] = drop_zeros( batch[key+'_active'], key+'_active', to_keep1 )
        new_batch[key+'2'] = drop_zeros( batch[key+'_inactive'], key+'_inactive', to_keep2 )
    # Define the atom masks
    atom_mask1 = new_batch['charges1'] > 0
    atom_mask2 = new_batch['charges2'] > 0
    new_batch['atom_mask1'] = atom_mask1
    new_batch['atom_mask2'] = atom_mask2
    # Define the edge masks
    edge_mask1 = atom_mask1.unsqueeze(1) * atom_mask1.unsqueeze(2)
    edge_mask2 = atom_mask2.unsqueeze(1) * atom_mask2.unsqueeze(2)
    new_batch['edge_mask1'] = edge_mask1
    new_batch['edge_mask2'] = edge_mask2
    return new_batch


def initialize_lep_data(args, datadir, splits = {'train':'train', 'valid':'val', 'test':'test'}):                        
    """
    Initialize datasets.

    :param args: Dictionary of input arguments detailing the cormorant calculation. 
    :type args: dict
    :param datadir: Path to the directory where the data and calculations and is, or will be, stored.
    :type datadir: str
    :param splits: Dictionary with sub-folder names for training, validation, and test set. Keys must be 'train', 'valid', 'test'.
    :type splits: dict, optional

    :return args: Dictionary of input arguments detailing the cormorant calculation.
    :rtype args: dict
    :return datasets: Dictionary of processed dataset objects. Valid keys are "train", "test", and "valid"
    :rtype datasets: dict
    :return num_species: Number of unique atomic species in the dataset.
    :rtype num_species: int
    :return max_charge: Largest atomic number for the dataset. 
    :rtype max_charge: pytorch.Tensor

    """
    # Define data files.
    datafiles = {split: os.path.join(datadir,splits[split]) for split in splits.keys()}
    # Load datasets
    datasets = _load_lep_data(datafiles, args.radius, args.droph, args.maxnum)
    # Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    _msg = 'Datasets must have the same set of keys!'
    assert all([key == keys[0] for key in keys]), _msg
    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets)
    # Now initialize the internal datasets based upon loaded data
    datasets = {split: CormorantDatasetLEP(data, included_species=all_species) for split, data in datasets.items()}
    # Check that all datasets have the same included species:
    _msg = 'All datasets must have the same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})
    assert (len(set(tuple(data.included_species.tolist()) for data in datasets.values())) == 1), _msg
    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge
    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts
    return args, datasets, num_species, max_charge


def _load_lep_data(datafiles, radius, droph, maxnum):
    """
    Load LEP datasets from LMDB format.

    :param datafiles: Dictionary of LMDB dataset directories.
    :type datafiles: dict
    :param radius: Radius of the selected region around the mutated residue.
    :type radius: float
    :param radius: Drop hydrogen atoms.
    :type radius: bool
    :param radius: Maximum number of atoms to consider.
    :type radius: int

    :return datasets: Dictionary of processed dataset objects.
    :rtype datasets: dict

    """
    datasets = {}
    key_names = ['index', 'num_atoms', 'charges', 'positions']
    for split, datafile in datafiles.items():
        dataset = LMDBDataset(datafile, transform=EnvironmentSelection(radius, droph, maxnum))
        # Load original atoms
        act = extract_coordinates_as_numpy_arrays(dataset, atom_frames=['atoms_active'])
        for k in key_names: act[k+'_active'] = act.pop(k)
        # Load mutated atoms
        ina = extract_coordinates_as_numpy_arrays(dataset, atom_frames=['atoms_inactive'])
        for k in key_names: ina[k+'_inactive'] = ina.pop(k)
        # Merge datasets with atoms
        dsdict = {**act, **ina}
        # Add labels (1 for active, 0 for inactive)
        ldict = {'A':1, 'I':0}
        labels = [ldict[dataset[i]['label']] for i in range(len(dataset))]
        dsdict['label'] = np.array(labels, dtype=int)
        # Convert everything to tensors
        datasets[split] = {key: torch.from_numpy(val) for key, val in dsdict.items()}
    return datasets


def _get_species(datasets, ignore_check=False):
    """
    Generate a list of all species.
    Includes a check that each split contains examples of every species in the entire dataset.
    
    :param datasets: Dictionary of datasets. Each dataset is a dict of arrays containing molecular properties.
    :type datasets: dict
    :param ignore_check: Ignores/overrides checks to make sure every split includes every species included in the entire dataset
    :type ignore_check: bool
    
    :return all_species: List of all species present in the data. Species labels should be integers.
    :rtype all_species: Pytorch tensor

    """
    # Find the unique list of species in each dataset.
    split_species = {split: torch.cat([ds['charges_active'].unique(),ds['charges_inactive'].unique()]).unique() for split, ds in datasets.items()}
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat( tuple(split_species.values()) ).unique()
    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0: all_species = all_species[1:]
    # Remove zeros if zero-padded charges exst for each split
    split_species = {split: species[1:] if species[0] == 0 else species for split, species in split_species.items()}
    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check: logging.error('The number of species is not the same in all datasets!')
        else: raise ValueError('Not all datasets have the same number of species!')
    # Finally, return a list of all species
    return all_species


class EnvironmentSelection(object):
    """
    Selects a region of protein coordinates within a certain distance from the alpha carbon of the mutated residue.

    :param df: Atoms data
    :type df: pandas.DataFrame
    :param dist: Distance from the alpha carbon of the mutated residue
    :type dist: float

    :return new_df: Transformed atoms data
    :rtype new_df: pandas.DataFrame

    """
    def __init__(self, dist, droph, maxnum):
        self._dist = dist
        self._droph = droph
        self._maxnum = maxnum

    def _drop_hydrogen(self, df):
        df_noh = df[df['element'] != 'H']
        print('Number of atoms after dropping hydrogen:', len(df_noh))
        return df_noh

    def _replace(self, df, keep=['H','C','N','O','S'], new='Cu'):
        new_elements = []
        for i in range(len(df['element'])):
            if df['element'][i] in keep:
                new_elements.append(df['element'][i])
            else:
                new_elements.append(new)
        df['element'] = new_elements
        return df

    def _select_env_by_dist(self, df, chain):
        # Separate pocket and ligand
        ligand = df[df['chain']==chain]
        pocket = df[df['chain']!=chain]
        # Extract coordinates
        ligand_coords = np.array([ligand.x, ligand.y, ligand.z]).T
        pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(pocket_coords)
        key_pts = kd_tree.query_ball_point(ligand_coords, r=self._dist, p=2.0)
        key_pts = np.unique([k for l in key_pts for k in l])
        # Construct the new data frame
        new_df = pd.concat([ pocket.iloc[key_pts], ligand ], ignore_index=True)
        print('Number of atoms after distance selection:', len(new_df))
        return new_df

    def _select_env_by_num(self, df, chain):
        # Separate pocket and ligand
        ligand = df[df['chain']==chain]
        pocket = df[df['chain']!=chain]
        # Max. number of protein atoms
        num = int(max([1, self._maxnum - len(ligand.x)]))
        # Extract coordinates
        ligand_coords = np.array([ligand.x, ligand.y, ligand.z]).T
        pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(pocket_coords)
        dd, ii = kd_tree.query(ligand_coords, k=len(pocket.x), p=2.0)
        # Get minimum distance to any lig atom for each protein atom
        dist = [ min(dd[ii==j]) for j in range(len(pocket.x)) ]
        # Sort indices by distance
        indices = np.argsort(dist)
        # Select the num closest atoms
        indices = np.sort(indices[:num])
        # Construct the new data frame
        new_df = pd.concat([ pocket.iloc[indices], ligand ], ignore_index=True)
        print('Number of atoms after number selection:', len(new_df))
        return new_df

    def __call__(self, x):
        # Select the ligand! 
        chain = 'L'
        # Replace rare atoms
        x['atoms_active'] = self._replace(x['atoms_active'])
        x['atoms_inactive'] = self._replace(x['atoms_inactive'])
        # Drop the hydrogen atoms
        if self._droph:
            x['atoms_active'] = self._drop_hydrogen(x['atoms_active'])
            x['atoms_inactive'] = self._drop_hydrogen(x['atoms_inactive'])
        # Select the environment
        x['atoms_active'] = self._select_env_by_dist(x['atoms_active'], chain)
        x['atoms_active'] = self._select_env_by_num(x['atoms_active'], chain)
        x['atoms_inactive'] = self._select_env_by_dist(x['atoms_inactive'], chain)
        x['atoms_inactive'] = self._select_env_by_num(x['atoms_inactive'], chain)
        return x

