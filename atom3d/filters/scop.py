"""
Filtering functions for protein structural classification by SCOP.

These all are applied to individual atom dataframes, and remove entries from that dataframe as necessary.
"""
import numpy as np
import pandas as pd

import atom3d.protein.scop as scop
import atom3d.util.file as fi


def form_scop_filter(level, allowed=None, excluded=None):
    """
    Create filter by SCOP classification at a specified level.

    Valid levels are: type, class, fold, superfamily, family.

    :param level: values that we will keep when they are found.
    :type level: str
    :param allowed: allowed SCOP values.
    :type allowed: list[str], optional
    :param excluded: excluded SCOP values.
    :type excluded: list[str], optional

    :return: function that implements the specified filter.
    :rtype: filter function
    """
    if excluded is not None and allowed is not None:
        raise RuntimeError('Can only specify one of allowed and excluded.')
    if allowed is None:
        allowed = []
    if excluded is None:
        excluded = []
    scop_index = scop.get_scop_index()
    scop_index = scop_index[level]

    # Build quick lookup tables.
    if allowed:
        permitted = pd.Series(
            {pdb_code: x.drop_duplicates().isin(allowed).any()
             for pdb_code, x in scop_index.groupby('pdb_code')})
    if excluded:
        forbidden = pd.Series(
            {pdb_code: x.drop_duplicates().isin(excluded).any()
             for pdb_code, x in scop_index.groupby('pdb_code')})

    def filter_fn(df):
        pdb_codes = df['structure'].apply(lambda x: x[:4].lower())

        if len(allowed) > 0:
            to_keep = permitted[pdb_codes]
            # If didn't find, be conservative and do not use.
            to_keep[to_keep.isna()] = False
            to_keep = to_keep.astype(bool)
        elif len(excluded) > 0:
            to_exclude = forbidden[pdb_codes]
            # If didn't find, be conservative and do not use.
            to_exclude[to_exclude.isna()] = True
            to_exclude = to_exclude.astype(bool)
            to_keep = ~to_exclude
        else:
            to_keep = pd.Series([True] * len(df), index=df['structure'])
        return df[to_keep.values]
    return filter_fn


def form_scop_filter_against(dataset, level, conservative):
    """
    Create filter that removes structures with matching SCOP class to a chain in supplied dataset.

    We consider each chain in each structure separately, and remove the structure if any of them matches any chain in dataset. This is done at the specified SCOP level.  Valid levels are: type, class, fold, superfamily, family.

    :param dataset: dataset that if we are checking for matches against.
    :type dataset: atom3d dataset
    :param level: SCOP level at which we are comparing datasets.
    :type level: str
    :param conservative: indicates what we should do about pdbs that do not have any SCOP class associated with them.  True means we throw out, False means we keep.
    :type conservative: bool

    :return: function that implements the specified filter.
    :rtype: filter function.
    """
    scop_index = scop.get_scop_index()[level]

    def form_scop_against():
        result = []
        for x in dataset:
            for (e, su, st), structure in x['atoms'].groupby(
                    ['ensemble', 'subunit', 'structure']):
                pc = fi.get_pdb_code(st).lower()
                for (m, c), _ in structure.groupby(['model', 'chain']):
                    if (pc, c) in scop_index:
                        result.append(scop_index.loc[(pc, c)].values)
        return np.unique(np.concatenate(result))
    scop_against = form_scop_against()

    def filter_fn(df):
        to_keep = {}
        for (e, su, st), structure in df.groupby(
                ['ensemble', 'subunit', 'structure']):
            pc = fi.get_pdb_code(st).lower()
            for (m, c), _ in structure.groupby(['model', 'chain']):
                if (pc, c) in scop_index:
                    scop_found = scop_index.loc[(pc, c)].values
                    if np.isin(scop_found, scop_against).any():
                        to_keep[(st, m, c)] = False
                    else:
                        to_keep[(st, m, c)] = True
                else:
                    to_keep[(st, m, c)] = not conservative
        to_keep = \
            pd.Series(to_keep)[pd.Index(df[['structure', 'model', 'chain']])]
        return df[to_keep.values]
    return filter_fn