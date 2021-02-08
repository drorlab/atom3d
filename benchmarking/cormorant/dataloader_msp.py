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


class CormorantDatasetMSP(Dataset):
    """
    Data structure for a Cormorant dataset. Extends PyTorch Dataset.

    :param data: Dictionary of arrays containing molecular properties.
    :type data: dict
    :param shuffle: If true, shuffle the points in the dataset.
    :type shuffle: bool, optional
        
    """
    def __init__(self, data, included_species=None, shuffle=False):
        # Define data and dataset size
        self.data = data
        self.num_pts = len(data['original_charges'])
        # If included species is not specified
        if included_species is None:
            all_charges = np.concatenate(self.data['original_charges'], self.data['mutated_charges'])
            self.included_species = torch.unique(all_charges, sorted=True)
        else:
            self.included_species = included_species
        # Convert charges to one-hot representation
        self.data['original_one_hot'] = self.data['original_charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        self.data['mutated_one_hot'] = self.data['mutated_charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        # Calculate parameters
        self.num_species = len(included_species)
        self.max_charge = max(included_species)
        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}
        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()
        if shuffle:
            self.perm = torch.randperm(len(data['original_charges']))[:self.num_pts]
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


def collate_msp(batch):
    """
    Collates MSP datapoints into the batch format for Cormorant.
    
    :param batch: The data to be collated.
    :type batch: list of datapoints

    :param batch: The collated data.
    :type batch: dict of Pytorch tensors

    """
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}
    # Define which fields to keep 
    to_keep1 = (batch['original_charges'].sum(0) > 0)
    to_keep2 = (batch['mutated_charges'].sum(0) > 0)
    # Start building the new batch
    new_batch = {}
    # Copy label data. 
    new_batch['label'] = batch['label']
    # Split structural data and drop zeros
    new_batch['charges1']   = drop_zeros( batch['original_charges'],   to_keep1 )
    new_batch['charges2']   = drop_zeros( batch['mutated_charges'],    to_keep2 )
    new_batch['positions1'] = drop_zeros( batch['original_positions'], to_keep1 )
    new_batch['positions2'] = drop_zeros( batch['mutated_positions'],  to_keep2 )
    new_batch['one_hot1']   = drop_zeros( batch['original_one_hot'],   to_keep1 )
    new_batch['one_hot2']   = drop_zeros( batch['mutated_one_hot'],    to_keep2 )
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


def initialize_msp_data(args, datadir, radius=6, splits = {'train':'train', 'valid':'val', 'test':'test'}):                        
    """
    Initialize datasets.

    :param args: Dictionary of input arguments detailing the cormorant calculation. 
    :type args: dict
    :param datadir: Path to the directory where the data and calculations and is, or will be, stored.
    :type datadir: str
    :param radius: Radius of the selected region around the mutated residue.
    :type radius: float
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
    datasets = _load_msp_data(datafiles, radius)
    # Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    _msg = 'Datasets must have same set of keys!'
    assert all([key == keys[0] for key in keys]), _msg
    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets)
    # Now initialize the internal datasets based upon loaded data
    datasets = {split: CormorantDatasetMSP(data, included_species=all_species) for split, data in datasets.items()}
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


def _load_msp_data(datafiles, radius):
    """
    Load MSP datasets from LMDB format.

    :param datafiles: Dictionary of LMDB dataset directories.
    :type datafiles: dict
    :param radius: Radius of the selected region around the mutated residue.
    :type radius: float

    :return datasets: Dictionary of processed dataset objects.
    :rtype datasets: dict

    """
    datasets = {}
    key_names = ['index', 'num_atoms', 'charges', 'positions']
    for split, datafile in datafiles.items():
        dataset = LMDBDataset(datafile, transform=EnvironmentSelection(radius))
        # Load original atoms
        ori = extract_coordinates_as_numpy_arrays(dataset, atom_frames=['original_atoms'])
        for k in key_names: ori['original_'+k] = ori.pop(k)
        # Load mutated atoms
        mut = extract_coordinates_as_numpy_arrays(dataset, atom_frames=['mutated_atoms'])
        for k in key_names: mut['mutated_'+k] = mut.pop(k)
        # Merge datasets with atoms
        datasets[split] = {**ori, **mut}
        # Add labels
        labels = [dataset[i]['label'] for i in range(len(dataset))]
        datasets[split]['label'] = np.array(labels, dtype=int)
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
    split_species = {split: np.array(np.unique(np.concatenate([ds['original_charges'],ds['mutated_charges']],axis=1)),dtype=int) for split, ds in datasets.items()}
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat( [torch.from_numpy(s) for s in split_species.values()] ).unique()
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
    def __init__(self, dist):
        self._dist = dist

    def _get_mutation(self, x):
        mutation = x['id'].split('_')[-1]
        chain = mutation[1]
        resid = int(mutation[2:-1])
        original_resname = mutation[0]
        mutation_resname = mutation[-1]
        return chain, resid

    def _select_env(self, df, chain, resid):
        # Find the C-alpha atom of the mutated residue
        mutated = df[(df.chain == chain) & (df.residue == resid)]
        mut_c_a = mutated[mutated.name == 'CA']
        # Define the protein atoms
        protein = df
        # extract coordinates
        muta_coords = np.array([mut_c_a.x, mut_c_a.y, mut_c_a.z]).T
        prot_coords = np.array([protein.x, protein.y, protein.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(prot_coords)
        key_pts = kd_tree.query_ball_point(muta_coords, r=self._dist, p=2.0)
        key_pts = np.unique([k for l in key_pts for k in l])
        # Construct the new data frame
        new_df = pd.concat([ protein.iloc[key_pts] ], ignore_index=True)
        return new_df

    def __call__(self, x):
        # Extract mutation info from the ID
        chain, resid = self._get_mutation(x)
        # Select environment in original data frame
        x['original_atoms'] = self._select_env(x['original_atoms'], chain, resid)
        # Select environment in mutated data frame
        x['mutated_atoms'] = self._select_env(x['mutated_atoms'], chain, resid)
        return x

