import atom3d.util.voxelize as vox

class GraphTransform(object):
    def __init__(self, atom_key, label_key):
        self.atom_key = atom_key
        self.label_key = label_key
    
    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        item = prot_graph_transform(item, atom_keys=[self.atom_key], label_key=self.label_key)
        
        return item[self.atom_key]
    
class PairedGraphTransform(object):
    def __init__(self, key_1, key_2, label_key):
        self.key_1 = key_1
        self.key_2 = key_2
        self.label_key = label_key
    
    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        item = prot_graph_transform(item, atom_keys=[self.key_1, self.key_2], label_key=self.label_key)
        
        return item[self.key_1], item[self.key_2]
    

def prot_graph_transform(item, atom_keys=['atoms'], label_key='scores'):
    """Transform for converting dataframes to Pytorch Geometric graphs, to be applied when defining a :mod:`Dataset <atom3d.datasets.datasets>`.
    Operates on Dataset items, assumes that the item contains all keys specified in ``keys`` and ``labels`` arguments.

    :param item: Dataset item to transform
    :type item: dict
    :param atom_keys: list of keys to transform, where each key contains a dataframe of atoms, defaults to ['atoms']
    :type atom_keys: list, optional
    :param label_key: name of key containing labels, defaults to ['scores']
    :type label_key: str, optional
    :return: Transformed Dataset item
    :rtype: dict
    """    
    from torch_geometric.data import Data
    import atom3d.util.graph as gr

    for key in atom_keys:
        node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(item[key])
        item[key] = Data(node_feats, edge_index, edge_feats, y=item[label_key], pos=pos)

    return item

def mol_graph_transform(item, atom_key='atoms', label_key='scores', allowable_atoms=None, use_bonds=False, onehot_edges=False):
    """Transform for converting dataframes to Pytorch Geometric graphs, to be applied when defining a :mod:`Dataset <atom3d.datasets.datasets>`.
    Operates on Dataset items, assumes that the item contains all keys specified in ``keys`` and ``labels`` arguments.

    :param item: Dataset item to transform
    :type item: dict
    :param atom_key: name of key containing molecule structure as a dataframe, defaults to 'atoms'
    :type atom_keys: list, optional
    :param label_key: name of key containing labels, defaults to 'scores'
    :type label_key: str, optional
    :param use_bonds: whether to use molecular bond information for edges instead of distance. Assumes bonds are stored under 'bonds' key, defaults to False
    :type use_bonds: bool, optional
    :return: Transformed Dataset item
    :rtype: dict
    """    
    from torch_geometric.data import Data
    import atom3d.util.graph as gr
    if use_bonds:
        bonds = item['bonds']
    else:
        bonds = None
    node_feats, edge_index, edge_feats, pos = gr.mol_df_to_graph(item[atom_key], bonds=bonds, onehot_edges=onehot_edges, allowable_atoms=allowable_atoms)
    item[atom_key] = Data(node_feats, edge_index, edge_feats, y=item[label_key], pos=pos)

    return item

def voxel_transform(item, grid_config, rot_mat=None, center_fn=vox.get_center, random_seed=None, structure_keys=['atoms']):
    """Transform for converting dataframes to voxelized grids compatible with 3D CNN, to be applied when defining a :mod:`Dataset <atom3d.datasets.datasets>`.
    Operates on Dataset items, assumes that the item contains all keys specified in ``keys`` argument.

    :param item: Dataset item to transform
    :type item: dict
    :param grid_config: Config parameters for grid. Should contain the following keys:
         `element_mapping`, dictionary mapping from element to 1-hot index; 
         `radius`, radius of grid to generate in Angstroms (half of side length); 
         `resolution`, voxel size in Angstroms;
         `num_directions`, number of directions for data augmentation (required if ``rot_mat``=None);
         `num_rolls`, number of rolls, or rotations, for data augmentation (required if ``rot_mat``=None);
    :type grid_config: :class:`dotdict <atom3d.util.voxelize.dotdict>`
    :param rot_mat: Rotation matrix (3x3) to apply to structure coordinates. If None (default), apply randomly sampled rotation according to parameters specified by ``grid_config.num_directions`` and ``grid_config.num_rolls``
    :type rot_mat: np.array
    :param center_fn: Arbitrary function for calculating the center of the voxelized grid (x,y,z coordinates) from a structure dataframe, defaults to vox.get_center
    :type center_fn: f(df -> array), optional
    :param random_seed: random seed for grid rotation, defaults to None
    :type random_seed: int, optional
    :return: Transformed Dataset item
    :rtype: dict
    """    
    
    for key in structure_keys:
        df = item[key]
        center = center_fn(df)

        if rot_mat is None:
            rot_mat = vox.gen_rot_matrix(grid_config, random_seed=random_seed)
        grid = vox.get_grid(
            df, center, config=grid_config, rot_mat=rot_mat)
        item[key] = grid
    return item
