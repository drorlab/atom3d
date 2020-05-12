import os

import collections as col


def get_target_name(filename):
    filename = str(filename)
    _, ext = os.path.splitext(filename)
    if ext in ['.dat', '.ss2']:
        return os.path.splitext(os.path.basename(filename))[0]
    elif ext in ['.pdb', '.mmcif']:
        return os.path.dirname(filename).split('/')[-1]
    else:
        raise ValueError("Unrecognized filetype {:}".format(filename))


def get_decoy_name(filename):
    filename = str(filename)
    _, ext = os.path.splitext(filename)
    if ext not in ['.pdb', '.mmcif', '.dat', '.ss2']:
        return os.path.basename(filename)
    else:
        return os.path.splitext(os.path.basename(filename))[0]


def split_by_target(filenames):
    ''' Split list of filenames by its targets. '''
    by_target_filenames = col.defaultdict(list)
    for filename in filenames:
        target_name = get_target_name(filename)
        by_target_filenames[target_name].append(filename)
    return by_target_filenames


def read_labels(labels_dir, ext='dat'):
    '''
    Read all label files with extension <ext> in <label_dir> into
    a panda DataFrame.
    '''
    files = fi.find_files(labels_dir, ext)
    frames = []
    for filename in files:
        target_name = get_target_name(filename)
        df = pd.read_csv(filename, delimiter='\s*', engine='python').dropna()
        frames.append(df)
    all_df = pd.concat(frames, sort=False).reset_index(drop=True)
    return all_df
