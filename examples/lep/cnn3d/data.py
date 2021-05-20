import os

import dotenv as de
import numpy as np
import pandas as pd
import torch

from atom3d.datasets import LMDBDataset
from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid
from torch.utils.data import DataLoader


de.load_dotenv(de.find_dotenv(usecwd=True))


class CNN3D_TransformLEP(object):
    def __init__(self, add_flag, random_seed=None, **kwargs):
        self.add_flag = add_flag
        self.random_seed = random_seed
        self.grid_config =  dotdict({
            # Mapping from elements to position in channel dimension.
            'element_mapping': {
                'H': 0,
                'C': 1,
                'O': 2,
                'N': 3,
                'S': 4,
                'CL': 5,
                'F': 6,
            },
            # Radius of the grids to generate, in angstroms.
            'radius': 25.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.0,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms, is_active):
        # Use center of ligand as subgrid center
        ligand_pos = atoms[atoms.chain == 'L'][['x', 'y', 'z']].astype(
            np.float32)
        ligand_center = get_center(ligand_pos)

        # Generate random rotation matrix
        rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
        # Transform protein/ligand into voxel grids and rotate
        grid = get_grid(atoms, ligand_center, config=self.grid_config, rot_mat=rot_mat)
        if self.add_flag:
            # Add inactive (0) or active (1) flag
            flag = np.full(grid.shape[:-1] + (1,), is_active)
            grid = np.concatenate([grid, flag], axis=3)

        # Last dimension is atom channel, so we need to move it to the front
        # per pytroch style
        grid = np.moveaxis(grid, -1, 0)
        return grid

    def __call__(self, item):
        # Transform structure into voxel grids.
        # Apply random rotation matrix.
        transformed = {
            'feature_inactive': self._voxelize(item['atoms_inactive'], False),
            'feature_active': self._voxelize(item['atoms_active'], True),
            'label': int(item['label'] == 'A'), # Convert to 0 for inactive, 1 for active
            'id': item['id']
        }
        return transformed


def create_balanced_sampler(dataset):
    # Assume labels/classes are integers (0, 1, 2, ...).
    labels = [item['label'] for item in dataset]
    classes, class_sample_count = np.unique(labels, return_counts=True)
    # Weighted sampler for imbalanced classification (1:1 ratio for each class)
    weight = 1. / class_sample_count
    sample_weights = torch.tensor([weight[t] for t in labels])
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights,
                                                     num_samples=len(dataset),
                                                     replacement=True)
    return sampler


if __name__=="__main__":
    dataset_path = os.path.join(os.environ['LEP_DATA'], 'val')
    dataset = LMDBDataset(
        dataset_path,
        transform=CNN3D_TransformLEP(add_flag=True, radius=10.0))
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        sampler=create_balanced_sampler(dataset))

    non_zeros = []
    for item in dataloader:
        print('feature inactive shape:', item['feature_inactive'].shape)
        print('feature active shape:', item['feature_active'].shape)
        print('label:', item['label'])
        print('id:', item['id'])
        non_zeros.append(np.count_nonzero(item['label']))
        break
    for item in dataloader:
        non_zeros.append(np.count_nonzero(item['label']))
    print(f"Count {sum(non_zeros):}/{len(dataset):} non-zero labels...")
    print(non_zeros)
