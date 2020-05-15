import numpy as  np
import pandas as pd

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
