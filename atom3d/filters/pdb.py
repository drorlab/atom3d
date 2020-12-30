"""
Filtering functions for protein data bank files.

These all are applied to individual atom dataframes, and remove entries from that dataframe as necessary.
"""
import pandas as pd

PDB_ENTRY_TYPE_FILE = 'atom3d/data/metadata/pdb_entry_type.txt'
RESOLUTION_FILE = 'atom3d/data/metadata/resolu.idx'


def form_size_filter(max_size=None, min_size=None):
    """
    Create filter for a certain number of residues, keeping only structures that fall within the specified bounds.

    :param max_size: maximum allowable number of residues in a structure.  None means there is no maximum.
    :type max_size: int.
    :param min_size: minimum allowable number of residues in a structure.  None means there is no minimum.
    :type min_size: int.

    :return: function that implements the specified filter.
    :rtype: filter function.
    """

    def filter_fn(df):
        to_keep = pd.Series([True] * len(df), index=df['structure'])
        residues = \
            df[['structure', 'model', 'chain', 'residue']].drop_duplicates()
        counts = residues['structure'].value_counts()
        if max_size:
            tmp = counts <= max_size
            tmp = tmp[df['structure']]
            to_keep = (to_keep & tmp)
        if min_size:
            tmp = counts >= min_size
            tmp = tmp[df['structure']]
            to_keep = (to_keep & tmp)
        return df[to_keep.values]
    return filter_fn


def form_source_filter(allowed=None, excluded=None):
    """
    Create filter for experimental source.

    Valid entries are: diffraction, NMR, EM, other.  Only one of allowed and excluded can be set.

    :param allowed: allowed experimental sources.
    :type allowed: list[str].
    :param excluded: excluded experimental sources.
    :type excluded: list[str].

    :return: function that implements the specified filter.
    :rtype: filter function.
    """
    if excluded is not None and allowed is not None:
        raise RuntimeError('Can only specify one of allowed and excluded.')
    if excluded is None:
        excluded = []
    if allowed is None:
        allowed = []
    pdb_entry_type = pd.read_csv(PDB_ENTRY_TYPE_FILE, delimiter='\t',
                                 names=['pdb_code', 'molecule_type', 'source'])
    pdb_entry_type['pdb_code'] = \
        pdb_entry_type['pdb_code'].apply(lambda x: x.lower())
    pdb_entry_type = pdb_entry_type.set_index('pdb_code', drop=True)
    source = pdb_entry_type['source']

    def filter_fn(df):
        pdb_codes = df['structure'].apply(lambda x: x[:4].lower())

        if len(allowed) > 0:
            to_keep = source[pdb_codes].isin(allowed)
        elif len(excluded) > 0:
            to_keep = ~source[pdb_codes].isin(excluded)
        else:
            to_keep = pd.Series([True] * len(df), index=df['structure'])
        return df[to_keep.values]
    return filter_fn


def form_molecule_type_filter(allowed=None, excluded=None):
    """
    Create filter for molecule type.

    Valid entries are: prot, prot-nuc, nuc, carb, other.  Only one of allowed and excluded can be set.

    :param allowed: allowed molecule types.
    :type allowed: list[str].
    :param excluded: excluded molecule types.
    :type excluded: list[str].

    :return: function that implements the specified filter.
    :rtype: filter function.
    """
    if allowed is None:
        allowed = []
    if excluded is None:
        excluded = []
    pdb_entry_type = pd.read_csv(PDB_ENTRY_TYPE_FILE, delimiter='\t',
                                 names=['pdb_code', 'molecule_type', 'source'])
    pdb_entry_type['pdb_code'] = \
        pdb_entry_type['pdb_code'].apply(lambda x: x.lower())
    pdb_entry_type = pdb_entry_type.set_index('pdb_code')
    molecule_type = pdb_entry_type['molecule_type']

    def filter_fn(df):
        pdb_codes = df['structure'].apply(lambda x: x[:4].lower())

        if len(allowed) > 0:
            to_keep = molecule_type[pdb_codes].isin(allowed)
        elif len(excluded) > 0:
            to_keep = ~molecule_type[pdb_codes].isin(excluded)
        else:
            to_keep = pd.Series([True] * len(df), index=df['structure'])
        return df[to_keep.values]
    return filter_fn


def form_resolution_filter(threshold):
    """
    Create filter for experimental resolution.

    :param threshold: maximum allowable experimental resolution.
    :type threshold: float.

    :return: function that implements the specified filter.
    :rtype: filter function.
    """
    resolution = pd.read_csv(RESOLUTION_FILE, skiprows=6, delimiter='\t',
                             usecols=[0, 2], names=['pdb_code', 'resolution'])
    resolution['pdb_code'] = resolution['pdb_code'].apply(lambda x: x.lower())
    resolution = resolution.set_index('pdb_code')['resolution']

    # Remove duplicates beyond the first.
    resolution = resolution[~resolution.index.duplicated()]

    def filter_fn(df):
        pdb_codes = df['structure'].apply(lambda x: x[:4].lower())
        to_keep = resolution[pdb_codes] < threshold
        return df[to_keep.values]
    return filter_fn
