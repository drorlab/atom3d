import os

import dotenv as de
import numpy as np
import pandas as pd

from atom3d.datasets import LMDBDataset
from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid
from torch.utils.data import DataLoader


de.load_dotenv(de.find_dotenv(usecwd=True))


class CNN3D_TransformSMP(object):
    def __init__(self, label_name, random_seed=None, **kwargs):
        self.label_name = label_name
        self.random_seed = random_seed
        self.grid_config =  dotdict({
            # Mapping from elements to position in channel dimension.
            'element_mapping': {
                'H': 0,
                'C': 1,
                'O': 2,
                'N': 3,
                'F': 4,
            },
            # Radius of the grids to generate, in angstroms.
            'radius': 7.5,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.0,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms):
        # Use center of molecule as subgrid center
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

    def _lookup_label(self, item, name):
        if 'label_mapping' not in self.__dict__:
            label_mapping = [
                'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
                'u0', 'u298', 'h298', 'g298', 'cv',
                'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom', 'cv_atom',
                ]
            self.label_mapping = {k: v for v, k in enumerate(label_mapping)}
        return item['labels'][self.label_mapping[name]]

    def __call__(self, item):
        # Transform molecule into voxel grids.
        # Apply random rotation matrix.
        transformed = {
            'feature': self._voxelize(item['atoms']),
            'label': self._lookup_label(item, self.label_name),
            'id': item['id'],
        }
        return transformed


if __name__=="__main__":
    dataset_path = os.path.join(os.environ['SMP_DATA'], 'val')
    dataset = LMDBDataset(
        dataset_path,
        transform=CNN3D_TransformSMP(label_name='alpha', radius=10.0))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    for item in dataloader:
        print('feature shape:', item['feature'].shape)
        print('label:', item['label'])
        print('id:', item['id'])
        break
