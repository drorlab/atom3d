"""Common filtering functions."""
import pandas as pd

import atom3d.util.scop as scop


PDB_ENTRY_TYPE_FILE = 'metadata/pdb_entry_type.txt'
RESOLUTION_FILE = 'metadata/resolu.idx'


def form_source_filter(allowed=[], excluded=[]):
    """
    Filter by experimental source.

    Valid entries are diffraction, NMR, EM, other.
    """
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


def form_molecule_type_filter(allowed=[], excluded=[]):
    """
    Filter by biomolecule type.

    Valid entries are prot, prot-nuc, nuc, carb, other.
    """
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
    """Filter by resolution of method used to determine."""
    resolution = pd.read_csv(RESOLUTION_FILE, skiprows=6, delimiter='\t',
                             usecols=[0, 2], names=['pdb_code', 'resolution'])
    resolution['pdb_code'] = resolution['pdb_code'].apply(lambda x: x.lower())
    resolution = resolution.set_index('pdb_code')['resolution']

    # Remove duplicates byoned the first.
    resolution = resolution[~resolution.index.duplicated()]

    def filter_fn(df):
        pdb_codes = df['structure'].apply(lambda x: x[:4].lower())
        to_keep = resolution[pdb_codes] < threshold
        return df[to_keep.values]
    return filter_fn


def form_size_filter(max_size=None, min_size=None):
    """Filter by number of residues."""

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


def first_model_filter(df):
    """Remove anything beyond first model in structure."""

    models = df[['structure', 'model']].drop_duplicates()
    models = models.sort_values(['structure', 'model'])

    models['to_keep'] = ~models['structure'].duplicated()
    models_to_keep = models.set_index(['structure', 'model'])

    to_keep = models_to_keep.loc[df.set_index(['structure', 'model']).index]
    return df[to_keep.values]


def form_scop_filter(level, allowed=[], excluded=[]):
    """
    Filter by SCOP classification at a specified level.

    Valid levels are type, class, fold, superfamily, family.
    """
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
