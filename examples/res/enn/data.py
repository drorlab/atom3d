import os
import logging
import numpy as np
import scipy as sp
import scipy.spatial
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from atom3d.datasets import LMDBDataset
from utils import batch_stack, drop_zeros


class CormorantDatasetRES(Dataset):
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
            all_charges = self.data['charges']
            self.included_species = torch.unique(all_charges, sorted=True)
        else:
            self.included_species = torch.unique(included_species, sorted=True)
        # Convert charges to one-hot representation
        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
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
        
        
def collate_res(batch):
    """
    Collates RES datapoints into the batch format for Cormorant.
    
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
    new_batch['label'] = batch['label']
    # Split structural data and drop zeros
    for key in ['charges','positions','one_hot']:
        new_batch[key] = drop_zeros( batch[key], key, to_keep )
    # Define the atom masks
    atom_mask = new_batch['charges'] > 0
    new_batch['atom_mask'] = atom_mask
    new_batch['atom_mask'] = atom_mask
    # Define the edge masks
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    new_batch['edge_mask'] = edge_mask
    return new_batch       


def initialize_res_data(args, datadir, splits = {'train':'train', 'valid':'val', 'test':'test'}):                        
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
    datasets = _load_res_data(datafiles) #, args.radius, args.droph, args.maxnum)
    # Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    _msg = 'Datasets must have the same set of keys!'
    assert all([key == keys[0] for key in keys]), _msg
    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets)
    # Now initialize the internal datasets based upon loaded data
    datasets = {split: CormorantDatasetRES(data, included_species=all_species) for split, data in datasets.items()}
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


def _load_res_data(datafiles):#, radius, droph, maxnum):
    """
    Load RES datasets from LMDB format.

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
        dataset = LMDBDataset(datafile)
        # Get labels 
        labels = np.concatenate([item['labels']['label'] for item in dataset])
        # Load original atoms
        dsdict = _extract_coordinates_as_numpy_arrays(dataset)
        for k in key_names: dsdict[k] = dsdict.pop(k)
        # Add labels
        dsdict['label'] = np.array(labels, dtype=int)
        # Convert everything to tensors
        datasets[split] = {key: torch.from_numpy(val) for key, val in dsdict.items()}
    return datasets


def _extract_coordinates_as_numpy_arrays(dataset, indices=None):
    """Convert the molecules from a dataset to a dictionary of numpy arrays.
       Labels are not processed; they are handled differently for every dataset.
    :param dataset: LMDB dataset from which to extract coordinates.
    :type dataset: torch.utils.data.Dataset
    :param indices: Indices of the items for which to extract coordinates.
    :type indices: numpy.array
    :return: Dictionary of numpy arrays with number of atoms, charges, and positions
    :rtype: dict
    """
    # Size of the dataset
    if indices is None:
        indices = np.arange(len(dataset), dtype=int)
    else:
        indices = np.array(indices, dtype=int)
        assert len(dataset) > max(indices)
    # Number of input items
    num_items = len(indices)

    # Calculate number of subunits and number of atoms for each subunit
    num_subunits = 0
    num_atoms = []
    for idx in indices:
        item = dataset[idx]
        num_subunits += len(item['subunit_indices'])
        for su in item['subunit_indices']:
            num_atoms.append(len(su)+1) # +1 for central atom
    num_atoms = np.array(num_atoms, dtype=int)

    # All charges and position arrays have the same size
    arr_size  = np.max(num_atoms)
    charges   = np.zeros([num_subunits,arr_size])
    positions = np.zeros([num_subunits,arr_size,3])
    # For each structure ...
    isu = 0
    for j,idx in enumerate(indices):
        item = dataset[idx]
        item['labels']['element'] = 'C'
        # .. and for each subunit ...
        for i,su in enumerate(item['subunit_indices']):
            # concatenate central atom and atoms from frame
            centr_at = item['labels'].iloc[[i]]
            su_atoms = item['atoms'].iloc[su]
            atoms = pd.concat([centr_at, su_atoms], ignore_index=True)
            # write per-atom data to arrays
            for ia in range(num_atoms[isu]):
                element = atoms['element'][ia].title()
                charges[isu,ia] = fo.atomic_number[element]
                positions[isu,ia,0] = atoms['x'][ia]
                positions[isu,ia,1] = atoms['y'][ia]
                positions[isu,ia,2] = atoms['z'][ia]
            isu += 1

    # Create a dictionary with all the arrays
    numpy_dict = {'index': indices, 'num_atoms': num_atoms,
                  'charges': charges, 'positions': positions}

    return numpy_dict
    

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

