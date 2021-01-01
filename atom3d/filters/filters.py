"""
Common filtering functions.

These all are applied to individual atom dataframes, and remove entries from that dataframe as necessary.
"""
import Bio.PDB.Polypeptide as Poly
import pandas as pd
import scipy.spatial as ss
import numpy as np


def standard_residue_filter(df):
    """
    Filter out non-standard residues.

    :param df: dataframe to filter against.
    :type df: atoms dataframe.

    :return: same dataframe, but with only with atoms corresponding to standard residues left.
    :rtype: atoms dataframe.
    """
    residues = df[['structure', 'model', 'chain', 'residue', 'resname']] \
        .drop_duplicates()
    sel = residues['resname'].apply(
        lambda x: Poly.is_aa(x, standard=True))

    residues['to_keep'] = sel
    residues_to_keep = residues.set_index(
        ['structure', 'model', 'chain', 'residue', 'resname'])['to_keep']
    to_keep = residues_to_keep.loc[df.set_index(
        ['structure', 'model', 'chain', 'residue', 'resname']).index]
    return df[to_keep.values]


def first_model_filter(df):
    """
    Remove anything beyond first model in structure.

    :param df: dataframe to filter against.
    :type df: atoms dataframe.

    :return: same dataframe, but with only with atoms corresponding to first model left.
    :rtype: atoms dataframe.
    """
    models = df[['structure', 'model']].drop_duplicates()
    models = models.sort_values(['structure', 'model'])

    models['to_keep'] = ~models['structure'].duplicated()
    models_to_keep = models.set_index(['structure', 'model'])

    to_keep = models_to_keep.loc[df.set_index(['structure', 'model']).index]
    return df[to_keep.values]


def first_chain_filter(df):
    """
    Remove anything beyond first model/chain in structure.

    :param df: dataframe to filter against.
    :type df: atoms dataframe.

    :return: same dataframe, but with only with atoms corresponding to the first chain of the first model left.
    :rtype: atoms dataframe.
    """

    chains = df[['structure', 'model', 'chain']].drop_duplicates()
    chains = chains.sort_values(['structure', 'model', 'chain'])

    chains['to_keep'] = ~chains['structure'].duplicated()
    chains_to_keep = chains.set_index(['structure', 'model', 'chain'])

    to_keep = \
        chains_to_keep.loc[df.set_index(['structure', 'model', 'chain']).index]
    return df[to_keep.values]

def single_chain_filter(df):
    """
    Remove anything beyond first model/chain in structure.

    :param df: dataframe to filter against.
    :type df: atoms dataframe.

    :return: same dataframe if structure has only one chain, otherwise empty dataframe.
    :rtype: atoms dataframe.
    """

    chains = df[['structure', 'model', 'chain']].drop_duplicates()
    chains = chains.sort_values(['structure', 'model', 'chain'])

    chains['to_keep'] = ~chains['structure'].duplicated(keep=False)
    chains_to_keep = chains.set_index(['structure', 'model', 'chain'])

    to_keep = \
        chains_to_keep.loc[df.set_index(['structure', 'model', 'chain']).index]
    return df[to_keep.values]

def identity_filter(df):
    """
    Leave atoms dataframe unchanged.

    :param df: dataframe to filter against.
    :type df: atoms dataframe.

    :return: same dataframe.
    :rtype: atoms dataframe.
    """
    return df

def distance_filter(df, pos, dist):
    """Returns dataframe containing all atoms within ``dist`` of query positions in ``pos``.

    :param df: Input structure dataframe
    :type df: pandas.DataFrame
    :param pos: x-y-z positions of query points. For N query points, should be array of shape N x 3
    :type pos: array-like
    :param dist: Distance in Angstrom to search for neighbors.
    :type dist: float
    :return: New dataframe containing only atoms within ``dist`` of query position
    :rtype: pandas.DataFrame
    """    
    if pos.ndim == 1:
        pos = pos[np.newaxis, :]
    kd_tree = ss.KDTree(df[['x','y','z']].values)
    nn_pt_idx = kd_tree.query_ball_point(pos, r=dist, p=2.0)
    nn_pt_idx = list(set([k for l in nn_pt_idx for k in l]))
    nn_df = df.iloc[nn_pt_idx].reset_index(drop=True)
    return nn_df

def compose(f, g):
    """
    Compose two filter f and g.

    :param f: Outer filter function.
    :type f: filter function.
    :param g: Inner filter function.
    :type g: filter function.

    :return: lambda x: f(g(x))
    :rtype: filter function.
    """

    def filter_fn(df):
        df = g(df)
        if len(df) > 0:
            return f(df)
        else:
            return df
    return filter_fn


def form_filter_against_list(against, column):
    """
    Create filter against a list, keeping only items with values in that list.

    :param against: values that we will keep when they are found.
    :type against: list.
    :param column: dataframe column that we will extract values from.
    :type column: str.

    :return: function that implements the specified filter.
    :rtype: filter function.
    """

    def filter_fn(df):
        to_keep = {}
        for e, ensemble in df.groupby([column]):
            to_keep[e] = e in against
        to_keep = pd.Series(to_keep)[df[column]]
        return df[to_keep.values]

    return filter_fn

def filter_to_transform(filter_fn, df_name='atoms'):
    """Create transform function (which operates on dataset items) from filter function (which operates on dataframes). By default, applies filter_fn to ``atoms`` dataframe, but a different dataframe can be specified optionally.

    :param filter_fn: Arbitrary filter function that takes in a dataframe and returns filtered dataframe
    :type filter_fn: filter function
    :param df_name: Name of dataframe to apply filter_fn to, defaults to 'atoms'
    :type df_name: str, optional
    """    
    def transform_fn(item):
        item[df_name] = filter_fn(item['atoms'])
        return item
    return transform_fn