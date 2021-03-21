import os
import copy
import logging
import numpy as np
import scipy as sp
import scipy.spatial
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from atom3d.datasets import LMDBDataset, extract_coordinates_as_numpy_arrays
from utils import batch_stack, drop_zeros


class CormorantDatasetLBA(Dataset):
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
        # Detect all charge keys
        charge_keys = []
        for key in self.data.keys():
            if 'charges' in key:
                charge_keys.append(key)
        # If included species is not specified
        if included_species is None:
            all_charges = np.concatenate([self.data[key] for key in charge_keys])
            self.included_species = torch.unique(all_charges, sorted=True)
        else:
            self.included_species = torch.unique(included_species, sorted=True)
        # Convert charges to one-hot representation
        for key in charge_keys:
            self.data[key.replace('charges','one_hot')] = self.data[key].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
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


def collate_lba(batch):
    """
    Collates LBA datapoints into the batch format for Cormorant.
    
    :param batch: The data to be collated.
    :type batch: list of datapoints

    :param batch: The collated data.
    :type batch: dict of Pytorch tensors

    """
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}
    # Define which fields to keep 
    to_keep = (batch['charges'].sum(0) > 0)
    # Start building the new batch
    new_batch = {}
    # Copy label data. 
    new_batch['neglog_aff'] = batch['neglog_aff']
    # Split structural data and drop zeros
    for key in ['charges','positions','one_hot']:
        new_batch[key] = drop_zeros( batch[key], key, to_keep )
    # Define the atom mask
    atom_mask = new_batch['charges'] > 0
    new_batch['atom_mask'] = atom_mask
    # Define the edge mask
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    new_batch['edge_mask'] = edge_mask
    return new_batch


def collate_lba_siamese(batch):
    """
    Collates LBA datapoints into the batch format for Cormorant.

    :param batch: The data to be collated.
    :type batch: list of datapoints

    :param batch: The collated data.
    :type batch: dict of Pytorch tensors

    """
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}
    # Define which fields to keep
    to_keep1 = (batch['bound_charges'].sum(0) > 0)
    to_keep2 = (batch['apart_charges'].sum(0) > 0)
    # Start building the new batch
    new_batch = {}
    # Copy label data.
    new_batch['neglog_aff'] = batch['neglog_aff']
    # Split structural data and drop zeros
    for key in ['charges','positions','one_hot']:
        new_batch[key+'1'] = drop_zeros( batch['bound_'+key], 'bound_'+key, to_keep1 )
        new_batch[key+'2'] = drop_zeros( batch['apart_'+key], 'apart_'+key, to_keep2 )
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


def initialize_lba_data(args, datadir, splits = {'train':'train', 'valid':'val', 'test':'test'}):                        
    """
    Initialize datasets.

    :param args: Dictionary of input arguments detailing the cormorant calculation. 
    :type args: dict
    :param datadir: Path to the directory where the data and calculations and is, or will be, stored.
    :type datadir: str
    :param radius: Radius of the selected region around the mutated residue.
    :type radius: float
    :param maxnum: Maximum total number of atoms of the ligand and the region around it.
    :type radius: int
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
    if args.siamese:
        datasets = _load_lba_data_siamese(datafiles, args.radius, args.maxnum)
    else:
        datasets = _load_lba_data(datafiles, args.radius, args.maxnum)
    # Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    _msg = 'Datasets must have same set of keys!'
    assert all([key == keys[0] for key in keys]), _msg
    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets)
    # Now initialize the internal datasets based upon loaded data
    datasets = {split: CormorantDatasetLBA(data, included_species=all_species) for split, data in datasets.items()}
    # Check that all datasets have the same included species:
    _msg = 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})
    assert (len(set(tuple(data.included_species.tolist()) for data in datasets.values())) == 1), _msg
    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge
    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts
    return args, datasets, num_species, max_charge


def _load_lba_data(datafiles, dist, maxnum):
    """
    Load LBA datasets from LMDB format.

    :param datafiles: Dictionary of LMDB dataset directories.
    :type datafiles: dict
    :param radius: Radius of the selected region around the ligand.
    :type radius: float
    :param maxnum: Maximum total number of atoms of the ligand and the region around it.
    :type radius: int

    :return datasets: Dictionary of processed dataset objects.
    :rtype datasets: dict

    """
    datasets = {}
    for split, datafile in datafiles.items():
        dataset = LMDBDataset(datafile, transform=TransformLBA(dist, maxnum, move_lig=False))
        # Load original atoms
        dsdict = extract_coordinates_as_numpy_arrays(dataset, atom_frames=['atoms_pocket','atoms_ligand'])
        # Add the label data
        dsdict['neglog_aff'] = np.array([item['scores']['neglog_aff'] for item in dataset])
        # Convert everything to tensors
        datasets[split] = {key: torch.from_numpy(val) for key, val in dsdict.items()}
    return datasets


def _load_lba_data_siamese(datafiles, dist, maxnum):
    """
    Load LBA datasets from LMDB format.

    :param datafiles: Dictionary of LMDB dataset directories.
    :type datafiles: dict
    :param radius: Radius of the selected region around the mutated residue.
    :type radius: float
    :param maxnum: Maximum total number of atoms of the ligand and the region around it.
    :type radius: int

    :return datasets: Dictionary of processed dataset objects.
    :rtype datasets: dict

    """
    datasets = {}
    key_names = ['index', 'num_atoms', 'charges', 'positions']
    for split, datafile in datafiles.items():
        dataset = LMDBDataset(datafile, transform=TransformLBA(dist, maxnum, move_lig=True))
        # Load original atoms
        bound = extract_coordinates_as_numpy_arrays(dataset, atom_frames=['atoms_pocket','atoms_ligand'])
        for k in key_names: bound['bound_'+k] = bound.pop(k)
        # Load mutated atoms
        apart = extract_coordinates_as_numpy_arrays(dataset, atom_frames=['atoms_pocket','atoms_ligand_moved'])
        for k in key_names: apart['apart_'+k] = apart.pop(k)
        # Merge datasets with atoms
        dsdict = {**bound, **apart}
        # Add the label data
        dsdict['neglog_aff'] = np.array([item['scores']['neglog_aff'] for item in dataset])
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
    split_species = {}
    for split, ds in datasets.items():
        si = []
        for key in ds.keys():
            if 'charges' in key: 
                si.append(ds[key].unique(sorted=True))
        split_species[split] = torch.cat(tuple(si)).unique(sorted=True)
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


class TransformLBA(object):

    def __init__(self, dist, maxnum, move_lig=True):
        self._dist = dist
        self._maxnum = maxnum
        self._dx = 0
        if move_lig: 
            self._dx = 1000

    def _move(self, df):
       df_moved = copy.deepcopy(df) 
       df_moved['x'] += self._dx
       return df_moved 

    def _drop_hydrogen(self, df):
        df_noh = df[df['element'] != 'H']
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

    def _select_env_by_dist(self, pocket, ligand):
        # Extract coordinates
        ligand_coords = np.array([ligand.x, ligand.y, ligand.z]).T
        pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(pocket_coords)
        key_pts = kd_tree.query_ball_point(ligand_coords, r=self._dist, p=2.0)
        key_pts = np.unique([k for l in key_pts for k in l])
        # Construct the new data frame
        new_pocket = pd.concat([ pocket.iloc[key_pts] ], ignore_index=True)
        return new_pocket

    def _select_env_by_num(self, pocket, ligand):
        # Max. number of protein atoms 
        num = int(max([1, self._maxnum - len(ligand.x)]))
        #print('Select a maximum of',num,'atoms.')
        # Extract coordinates
        ligand_coords = np.array([ligand.x, ligand.y, ligand.z]).T
        pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(pocket_coords)
        dd, ii = kd_tree.query(ligand_coords, k=len(pocket.x), p=2.0)
        # Get minimum distance to any lig atom for each protein atom
        dis = [ min(dd[ii==j]) for j in range(len(pocket.x)) ]
        # Sort indices by distance
        idx = np.argsort(dis)
        # Select the num closest atoms
        idx = np.sort(idx[:num])
        # Construct the new data frame
        new_pocket = pd.concat([ pocket.iloc[idx] ], ignore_index=True)
        return new_pocket

    def __call__(self, x):
        # Do not use the full protein
        x['atoms_protein'] = None
        # Replace rare atoms
        x['atoms_pocket'] = self._replace(x['atoms_pocket'])
        x['atoms_ligand'] = self._replace(x['atoms_ligand'])
        # Drop hydrogen atoms
        x['atoms_pocket'] = self._drop_hydrogen(x['atoms_pocket'])
        x['atoms_ligand'] = self._drop_hydrogen(x['atoms_ligand'])
        # Select the environment
        x['atoms_pocket'] = self._select_env_by_dist(x['atoms_pocket'],x['atoms_ligand'])
        x['atoms_pocket'] = self._select_env_by_num(x['atoms_pocket'],x['atoms_ligand'])
        # Move the ligand far away 
        if self._dx != 0: 
            x['atoms_ligand_moved'] = self._move(x['atoms_ligand'])
        return x


