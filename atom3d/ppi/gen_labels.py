"""Methods to extract protein interface labels from pdb file[s]."""
import os

import click
import numpy as np
import pandas as pd
import scipy.spatial as spa

import atom3d.util.datatypes as dt


index_columns = ['structure', 'model', 'chain', 'residue']


@click.command()
@click.argument('input_pdbs', nargs=-1, type=click.Path(exists=True))
@click.argument('output_labels', type=click.Path())
@click.option('-b', '--bound_pdbs', multiple=True,
              type=click.Path(exists=True))
@click.option('-c', '--cutoff', type=int, default=8)
@click.option('--cutoff-type', default='CA',
              type=click.Choice(['heavy', 'CA'], case_sensitive=False))
def gen_labels_main(*args, **kwargs):
    gen_labels(*args, **kwargs)


def gen_labels(input_pdbs, output_labels, bound_pdbs, cutoff, cutoff_type):
    """Given input pdbs, and optionally bound pdbs, generate label file."""
    dfs = []
    for input_pdb in input_pdbs:
        bp = dt.read_pdb(input_pdb)
        dfs.append(dt.bp_to_df(bp))

    if len(bound_pdbs) > 0:
        if len(bound_pdbs) != len(input_pdbs):
            raise RuntimeError('If providing bound pdbs, provide same number '
                               'and in same order as input pdbs.')
        bound_dfs = []
        for bound_pdb in bound_pdbs:
            bp = dt.read_pdb(bound_pdb)
            bound_dfs.append(dt.bp_to_df(bp))

        subunits = bound_dfs
    else:
        df = pd.concat(dfs)
        subunits = [x for _, x in df.groupby(['structure', 'model', 'chain'])]

    neighbors = []
    for i in range(len(subunits)):
        for j in range(i + 1, len(subunits)):
            if cutoff_type == 'CA':
                curr = _get_ca_neighbors(subunits[i], subunits[j], cutoff)
            else:
                curr = _get_heavy_neighbors(subunits[i], subunits[j], cutoff)
            neighbors.append(curr)

    neighbors = pd.concat(neighbors)
    neighbors['label'] = 1

    if len(bound_pdbs) > 0:
        correspondence = {}
        for b, i in zip(bound_pdbs, input_pdbs):
            correspondence[os.path.basename(b)] = os.path.basename(i)

        neighbors['structure0'] = \
            neighbors['structure0'].apply(lambda x: correspondence[x])
        neighbors['structure1'] = \
            neighbors['structure1'].apply(lambda x: correspondence[x])

    neighbors.to_csv(output_labels, index=False)


def get_all_res(df):
    return df[index_columns].drop_duplicates()


def _get_ca_neighbors(df0, df1, cutoff):
    """Get neighbors for alpha-carbon based distance."""
    ca0 = df0[df0['atom_name'] == 'CA']
    ca1 = df1[df1['atom_name'] == 'CA']

    dist = spa.distance.cdist(ca0[['x', 'y', 'z']], ca1[['x', 'y', 'z']])
    pairs = np.array(np.where(dist < cutoff)).T
    res0 = ca0.iloc[pairs[:, 0]][index_columns]
    res1 = ca1.iloc[pairs[:, 1]][index_columns]
    res0 = res0.reset_index(drop=True).add_suffix('0')
    res1 = res1.reset_index(drop=True).add_suffix('1')
    res = pd.concat((res0, res1), axis=1)
    return res


def _get_heavy_neighbors(df0, df1, cutoff):
    """Get neighbors for heavy atom based distance."""
    heavy0 = df0[df0['element'] != 'H']
    heavy1 = df1[df1['element'] != 'H']

    dist = spa.distance.cdist(heavy0[['x', 'y', 'z']], heavy1[['x', 'y', 'z']])
    pairs = np.array(np.where(dist < cutoff)).T
    res0 = heavy0.iloc[pairs[:, 0]][index_columns]
    res1 = heavy1.iloc[pairs[:, 1]][index_columns]
    res0 = res0.reset_index(drop=True).add_suffix('0')
    res1 = res1.reset_index(drop=True).add_suffix('1')
    # We concatenate and find unique _pairs_.
    res = pd.concat((res0, res1), axis=1)
    res = res.drop_duplicates()
    return res


if __name__ == "__main__":
    gen_labels_main()
