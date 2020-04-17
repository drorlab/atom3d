import os

import collections as col


def get_target_name(filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.dat']:
        return os.path.splitext(os.path.basename(filename))[0]
    elif ext in ['.pdb', '.mmcif']:
        return os.path.dirname(filename).split('/')[-1]
    else:
        raise ValueError("Unrecognized filetype {:}".format(filename))


def get_decoy_name(filename):
    _, ext = os.path.splitext(filename)
    if ext not in ['.pdb', '.mmcif', '.dat']:
        return os.path.basename(filename)
    else:
        return os.path.splitext(os.path.basename(filename))[0]


def split_by_target(filenames):
    ''' Split list of filenames by its targets '''
    by_target_filenames = col.defaultdict(list)
    for filename in filenames:
        target_name = get_target_name(filename)
        by_target_filenames[target_name].append(filename)
    return by_target_filenames