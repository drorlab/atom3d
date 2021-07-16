import numpy as np
import pandas as pd

import atom3d.util.vectors as vec

import itertools
import operator

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_center(pos):
    """ Return the center coordinate. """
    return pos.mean().to_numpy().astype(np.float32)


def get_max_distance_from_center(pos, center):
    """ Return maximum distance from center. """
    max_dist = np.sqrt(pos.sub(center).pow(2).sum(1).max())
    return max_dist


def distribute(sequence):
    """
    Enumerate the sequence evenly over the interval (0, 1).

    >>> list(distribute('abc'))
    [(0.25, 'a'), (0.5, 'b'), (0.75, 'c')]
    """
    m = len(sequence) + 1
    for i, x in enumerate(sequence, 1):
        yield i/m, x


def intersperse(*sequences):
    """
    Evenly intersperse the sequences.

    Based on https://stackoverflow.com/a/19293603/4518341

    >>> list(intersperse(range(10), 'abc'))
    [0, 1, 'a', 2, 3, 4, 'b', 5, 6, 7, 'c', 8, 9]
    >>> list(intersperse('XY', range(10), 'abc'))
    [0, 1, 'a', 2, 'X', 3, 4, 'b', 5, 6, 'Y', 7, 'c', 8, 9]
    >>> ''.join(intersperse('hlwl', 'eood', 'l r!'))
    'hello world!'
    """
    distributions = map(distribute, sequences)
    get0 = operator.itemgetter(0)
    for _, x in sorted(itertools.chain(*distributions), key=get0):
        yield x


def rot_mats(config):
    # Generate the rotation matrices.
    if '_rot_mats' not in rot_mats.__dict__:
        _uvs = vec.fibonacci(config.num_directions)
        _ups = vec.generate_all_up_vectors(_uvs, config.num_rolls)
        rot_mats._rot_mats = vec.get_all_rot_mats(_uvs, _ups).astype(np.float32)
    return rot_mats._rot_mats


def gen_rot_matrix(config, random_seed=None, reset_seed=False):
    if reset_seed:
        np.random.seed(random_seed)

    size = grid_size(config)
    true_radius = size * config.resolution / 2.0
    direction = np.random.randint(low=0, high=config.num_directions, dtype=np.int32)
    roll = np.random.randint(low=0, high=config.num_rolls, dtype=np.int32)
    rot_mat = rot_mats(config)[direction][roll]
    return rot_mat


def get_grid(df, center, config, rot_mat=np.eye(3, 3)):
    """
    Generate the 3d grid from coordinate format.
    Args:
        df (pd.DataFrame):
            region to generate grid for.
        center (3x3 np.array):
            center of the grid.
        rot_mat (3x3 np.array):
            rotation matrix to apply to region before putting in grid.
    Returns:
        4-d numpy array representing an occupancy grid where last dimension
        is atom channel.  First 3 dimension are of size radius_ang * 2 + 1.
    """
    size = grid_size(config)
    true_radius = size * config.resolution / 2.0

    at = df[['x', 'y', 'z']].values.astype(np.float32)
    elements = df['element'].values

    # Center atoms.
    at = at - center

    # Apply rotation matrix.
    at = np.dot(at, rot_mat)
    at = (np.around((at + true_radius) / config.resolution - 0.5)).astype(np.int16)

    # Prune out atoms outside of grid as well as non-existent atoms.
    sel = np.all(at >= 0, axis=1) & np.all(at < size, axis=1) & (elements != '')
    at = at[sel]

    # Select valid atoms and form final grid.
    labels = elements[sel]
    lsel = np.nonzero([_recognized(x, config.element_mapping) for x in labels])
    labels = labels[lsel]
    labels = np.array([config.element_mapping[x] for x in labels], dtype=np.int8)

    grid = np.zeros(grid_shape(config), dtype=np.float32)
    grid[at[lsel, 0], at[lsel, 1], at[lsel, 2], labels] = 1

    return grid


def _recognized(x, dict):
    """ If atom type is recognized, return it.  Else, return empty string. """
    if x in dict.keys():
        return x
    else:
        return ''


def num_channels(config):
    # Multiple elements may be mapped to the same encoding, so we 
    # need to take the max + 1 (assume encoding starts from 0)
    return max(config.element_mapping.values()) + 1


def num_element_types(config):
    return len(config.element_mapping)


def grid_size(config):
    """ Get size of grid. """
    return int(round((config.radius * 2 + 1) / config.resolution))


def grid_shape(config):
    """ Return shape of grid. """
    size = grid_size(config)
    return (size, size, size, num_channels(config))
