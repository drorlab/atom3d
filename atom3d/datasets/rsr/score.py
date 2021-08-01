"""Code for loading rosetta score files."""
import os
import re

import pandas as pd

import atom3d.util.file as fi


def form_score_filter(score_dir):
    """Filter by decoys present in score file."""
    scores = load_scores(score_dir)

    def is_present(x):
        if x['ensemble'] not in scores:
            return False
        if x['subunit'] not in scores[x['ensemble']].index:
            return False
        return True

    def filter_fn(df):
        scores2 = scores.copy()
        ensembles = df[['ensemble', 'subunit']].drop_duplicates()
        ensembles['to_keep'] = ensembles.apply(is_present, axis=1)
        ensembles_to_keep = ensembles.set_index(['ensemble', 'subunit'])
        to_keep = ensembles_to_keep.loc[
            df.set_index(['ensemble', 'subunit']).index]
        return df[to_keep.values]

    return filter_fn


NUMBER_PATTERN = re.compile('_([0-9]{1,2})(_|$|\.)')


def get_target_number(f):
    """Extract integer target number from FARFAR2 file."""
    return int(re.search(NUMBER_PATTERN, f).group(1))


def get_target_name(f):
    """Extract string target name from FARFAR2 file."""
    name = os.path.basename(f)

    target_number = get_target_number(name)
    if target_number != 14:
        target_name = str(target_number)
    else:
        # We keep bound and free denotation if puzzle 14.
        target_name = str(target_number) + \
            ('b' if 'bound' in name else 'f')
    return target_name


def load_scores(score_dir):
    """Create target_name -> (subunit_name -> RMS)."""
    score_files = fi.find_files(score_dir, 'sc')
    scores = {
        get_target_name(f):
        pd.read_csv(f, delimiter='\s*', index_col='description',
                    engine='python')
        for f in score_files
    }
    # If duplicate structures present, remove all but first.
    for x, y in scores.items():
        scores[x] = y.loc[~y.index.duplicated(keep='first')]
    return scores
