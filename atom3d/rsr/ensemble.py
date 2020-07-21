"""Code to ensemble FARFAR2-Puzzles dataset."""
import os
import random
import re

import collections as col


NUMBER_PATTERN = re.compile('_([0-9]{1,2})(_|$|\.)')


def get_target_number(f):
    """Extract integer target number from FARFAR2 file."""
    return int(re.search(NUMBER_PATTERN, f).group(1))


def get_target_name(f):
    """Extract string target name from FARFAR2 file."""
    dir_name = os.path.basename(os.path.dirname(f))

    target_number = get_target_number(dir_name)
    if target_number != 14:
        target_name = str(target_number)
    else:
        # We keep bound and free denotation if puzzle 14.
        target_name = str(target_number) + \
            ('b' if 'bound' in dir_name else 'f')
    return target_name


def get_decoy_name(f):
    """Extract name of specific structural model."""
    return os.path.splitext(os.path.basename(f))[0]


def rsr_ensembler(pdb_files):
    ensembles = col.defaultdict(dict)
    random.shuffle(pdb_files)

    for f in pdb_files:
        target_name = get_target_name(f)
        if len(ensembles[target_name]) >= 1000:
            # Only graph
            continue
        decoy_name = get_decoy_name(f)
        ensembles[target_name][decoy_name] = f

    return ensembles
