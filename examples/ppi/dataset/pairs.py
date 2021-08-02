"""Code to generate pair ensembles."""
import click
import pandas as pd
import parallel as par

import neighbors as nb
import atom3d.util.log as log

logger = log.get_logger('shard_pairs')


def _gen_pairs_per_ensemble(x, cutoff, cutoff_type):
    pairs = []
    if len(x['subunit'].unique()) > 1:
        raise RuntimeError('Cannot find pairs on existing ensemble')
    # Only keep first model.
    x = x[x['model'] == sorted(x['model'].unique())[0]]
    names, subunits = _gen_subunits(x)

    for i in range(len(subunits)):
        for j in range(i + 1, len(subunits)):
            curr = nb.get_neighbors(
                subunits[i], subunits[j], cutoff, cutoff_type)
            if len(curr) > 0:
                tmp0 = subunits[i].copy()
                tmp0['subunit'] = names[i]
                tmp1 = subunits[j].copy()
                tmp1['subunit'] = names[j]
                pair = pd.concat([tmp0, tmp1])
                pair['ensemble'] = names[i] + '_' + names[j]
                pairs.append(pair)
    return pairs


def _gen_subunits(df):
    """Extract subunits to define protein interfaces for."""
    names = []

    subunits = []
    for name, x in df.groupby(['structure', 'model', 'chain']):
        names.append('_'.join([str(x) for x in name]))
        subunits.append(x)
    return names, subunits
