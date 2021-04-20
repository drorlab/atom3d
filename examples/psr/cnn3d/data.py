import os

import dotenv as de
import numpy as np
import pandas as pd

from atom3d.datasets import LMDBDataset
from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid
from torch.utils.data import DataLoader


de.load_dotenv(de.find_dotenv(usecwd=True))


class CNN3D_TransformPSR(object):
    def __init__(self, random_seed=None, **kwargs):
        self.random_seed = random_seed
        self.grid_config =  dotdict({
            # Mapping from elements to position in channel dimension.
            'element_mapping': {
                'C': 0,
                'O': 1,
                'N': 2,
                'S': 3,
            },
            # Radius of the grids to generate, in angstroms.
            'radius': 40.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.3,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms):
        # Use center of protein as subgrid center
        pos = atoms[['x', 'y', 'z']].astype(np.float32)
        center = get_center(pos)
        # Generate random rotation matrix
        rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
        # Transform protein/ligand into voxel grids and rotate
        grid = get_grid(atoms, center, config=self.grid_config, rot_mat=rot_mat)
        # Last dimension is atom channel, so we need to move it to the front
        # per pytroch style
        grid = np.moveaxis(grid, -1, 0)
        return grid

    def __call__(self, item):
        # Transform protein into voxel grids.
        # Apply random rotation matrix.
        id = eval(item['id'])
        transformed = {
            'feature': self._voxelize(item['atoms']),
            'label': item['scores']['gdt_ts'],
            'target': id[0],
            'decoy': id[1],
        }
        return transformed


if __name__=="__main__":
    dataset_path = os.path.join(os.environ['PSR_DATA'], 'val')
    dataset = LMDBDataset(dataset_path, transform=CNN3D_TransformPSR(radius=10.0))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    for item in dataloader:
        print('feature shape:', item['feature'].shape)
        print('label:', item['label'])
        print('target:', item['target'])
        print('decoy:', item['decoy'])
        break
