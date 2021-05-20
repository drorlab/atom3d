import math
import os
import torch

import dotenv as de
import numpy as np
import pandas as pd

from atom3d.datasets import LMDBDataset
from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid, intersperse
from torch.utils.data import DataLoader, IterableDataset

import atom3d.datasets.ppi.neighbors as nb


de.load_dotenv(de.find_dotenv(usecwd=True))


class CNN3D_Dataset(IterableDataset):
    def __init__(self, lmdb_path, testing, random_seed=None, **kwargs):
        self._lmdb_dataset = LMDBDataset(lmdb_path)
        self.testing = testing
        self.random_seed = random_seed
        self.grid_config =  dotdict({
            # Mapping from elements to position in channel dimension.
            'element_mapping': {
                'C': 0,
                'O': 1,
                'N': 2,
                'S': 3
            },
            # Radius of the grids to generate, in angstroms.
            'radius': 17.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.0,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,

            ### PPI specific
            # Number of negatives to sample per positive example. -1 means all.
            'neg_to_pos_ratio': 1,
            'neg_to_pos_ratio_testing': 1,
            # Max number of positive regions to take from a structure. -1 means all.
            'max_pos_regions_per_ensemble': 5,
            'max_pos_regions_per_ensemble_testing': 5,
            # Whether to use all negative at test time.
            'full_test': False,
        })
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def __len__(self) -> int:
        return len(self._lmdb_dataset)

    def _get_voxel_center(self, df, mut_chain, mut_res):
        if self.center_at_mut:
            # Use CA position of the mutated residue as grid center
            sel = ((df.chain == mut_chain) &
                   (df.residue == mut_res) &
                   (df.name == 'CA'))
            pos = df[sel][['x', 'y', 'z']].astype(np.float32)
        else:
            pos = df[['x', 'y', 'z']].astype(np.float32)
        return get_center(pos)

    def _voxelize(self, struct0, struct1, center0, center1):
        def _feature(struct, center):
            # Generate random rotation matrix
            rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
            # Transform into voxel grids and rotate
            grid = get_grid(
                struct, center, config=self.grid_config, rot_mat=rot_mat)
            # Last dimension is atom channel, so we need to move it to the front
            # per pytroch style
            grid = np.moveaxis(grid, -1, 0)
            return grid

        grid0 = _feature(struct0, center0)
        grid1 = _feature(struct1, center1)
        return grid0, grid1

    def _num_to_use(self, num_pos, num_neg):
        if self.testing:
            neg_to_pos_ratio = self.grid_config.neg_to_pos_ratio_testing
            max_pos_regions_per_ensemble = self.grid_config.max_pos_regions_per_ensemble_testing
        else:
            neg_to_pos_ratio = self.grid_config.neg_to_pos_ratio
            max_pos_regions_per_ensemble = self.grid_config.max_pos_regions_per_ensemble

        if neg_to_pos_ratio == -1 or (self.testing and self.grid_config.full_test):
            num_pos_to_use, num_neg_to_use = num_pos, num_neg
        else:
            num_pos_to_use = min(num_pos, num_neg / neg_to_pos_ratio)
            if (max_pos_regions_per_ensemble != -1):
                num_pos_to_use = min(num_pos_to_use, max_pos_regions_per_ensemble)
            num_neg_to_use = num_pos_to_use * neg_to_pos_ratio
        num_pos_to_use = int(math.ceil(num_pos_to_use))
        num_neg_to_use = int(math.ceil(num_neg_to_use))
        return num_pos_to_use, num_neg_to_use

    def _get_res_pair_ca_coords(self, samples_df, structs_df):
        def _get_ca_coord(struct, res):
            coord = struct[(struct.residue == res) & (struct.name == 'CA')][['x', 'y', 'z']].values[0]
            return coord

        res_pairs = samples_df[['residue0', 'residue1']].values
        cas = []
        for (res0, res1) in res_pairs:
            try:
                coord0 = _get_ca_coord(structs_df[0], res0)
                coord1 = _get_ca_coord(structs_df[1], res1)
                cas.append((res0, res1, coord0, coord1))
            except:
                pass
        return cas

    def __iter__(self):
        for index in range(len(self._lmdb_dataset)):
            item = self._lmdb_dataset[index]

            # Subunits
            names, (bdf0, bdf1, udf0, udf1) = nb.get_subunits(item['atoms_pairs'])
            structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
            # Get positives
            pos_neighbors_df = item['atoms_neighbors']
            # Get negatives
            neg_neighbors_df = nb.get_negatives(pos_neighbors_df, structs_df[0], structs_df[1])

            # Throw away non empty hetero/insertion_code
            non_heteros = []
            for df in structs_df:
                non_heteros.append(df[(df.hetero==' ') & (df.insertion_code==' ')].residue.unique())
            pos_neighbors_df = pos_neighbors_df[pos_neighbors_df.residue0.isin(non_heteros[0]) & \
                                                pos_neighbors_df.residue1.isin(non_heteros[1])]
            neg_neighbors_df = neg_neighbors_df[neg_neighbors_df.residue0.isin(non_heteros[0]) & \
                                                neg_neighbors_df.residue1.isin(non_heteros[1])]

            # Sample pos and neg samples
            num_pos = pos_neighbors_df.shape[0]
            num_neg = neg_neighbors_df.shape[0]
            num_pos_to_use, num_neg_to_use = self._num_to_use(num_pos, num_neg)

            if pos_neighbors_df.shape[0] == num_pos_to_use:
                pos_samples_df = pos_neighbors_df.reset_index(drop=True)
            else:
                pos_samples_df = pos_neighbors_df.sample(num_pos_to_use, replace=True).reset_index(drop=True)
            if neg_neighbors_df.shape[0] == num_neg_to_use:
                neg_samples_df = neg_neighbors_df.reset_index(drop=True)
            else:
                neg_samples_df = neg_neighbors_df.sample(num_neg_to_use, replace=True).reset_index(drop=True)

            pos_pairs_cas = self._get_res_pair_ca_coords(pos_samples_df, structs_df)
            neg_pairs_cas = self._get_res_pair_ca_coords(neg_samples_df, structs_df)

            pos_features = []
            for (res0, res1, center0, center1) in pos_pairs_cas:
                grid0, grid1 = self._voxelize(structs_df[0], structs_df[1], center0, center1)
                pos_features.append({
                    'feature_left': grid0,
                    'feature_right': grid1,
                    'label': 1,
                    'id': '{:}/{:}/{:}'.format(item['id'], res0, res1),
                })
            neg_features = []
            for (res0, res1, center0, center1) in neg_pairs_cas:
                grid0, grid1 = self._voxelize(structs_df[0], structs_df[1], center0, center1)
                neg_features.append({
                    'feature_left': grid0,
                    'feature_right': grid1,
                    'label': 0,
                    'id': '{:}/{:}/{:}'.format(item['id'], res0, res1),
                })
            for f in intersperse(pos_features, neg_features):
                yield f


if __name__=="__main__":
    dataset_path = os.path.join(os.environ['PPI_DATA'], 'test')
    dataset = CNN3D_Dataset(dataset_path, testing=False, radius=10.0)
    dataloader = DataLoader(
        dataset,
        batch_size=8)

    non_zeros = []
    for item in dataloader:
        print('feature left shape:', item['feature_left'].shape)
        print('feature right shape:', item['feature_right'].shape)
        print('label:', item['label'])
        print('id:', item['id'])
        non_zeros.append(np.count_nonzero(item['label']))
        break
