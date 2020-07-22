"""Common filtering functions."""
import Bio.PDB.Polypeptide as Poly
import pandas as pd


def standard_residue_filter(df):
    """Filter out non-standard residues."""
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
    """Remove anything beyond first model in structure."""

    models = df[['structure', 'model']].drop_duplicates()
    models = models.sort_values(['structure', 'model'])

    models['to_keep'] = ~models['structure'].duplicated()
    models_to_keep = models.set_index(['structure', 'model'])

    to_keep = models_to_keep.loc[df.set_index(['structure', 'model']).index]
    return df[to_keep.values]


def single_chain_filter(df):
    """Remove anything that has more than one model/chain."""

    chains = df[['structure', 'model', 'chain']].drop_duplicates()
    chains = chains.sort_values(['structure', 'model', 'chain'])

    chains['to_keep'] = ~chains['structure'].duplicated(False)
    chains_to_keep = chains.set_index(['structure', 'model', 'chain'])

    to_keep = \
        chains_to_keep.loc[df.set_index(['structure', 'model', 'chain']).index]
    return df[to_keep.values]


def identity_filter(df):
    return df


def compose(f, g):

    def filter_fn(df):
        df = g(df)
        if len(df) > 0:
            return f(df)
        else:
            return df
    return filter_fn


def form_filter_against_list(against, column):
    """Filter against list for column of dataframe."""

    def filter_fn(df):
        to_keep = {}
        for e, ensemble in df.groupby([column]):
            to_keep[e] = e in against
        to_keep = pd.Series(to_keep)[df[column]]
        return df[to_keep.values]

    return filter_fn
