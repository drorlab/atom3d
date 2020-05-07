"""Methods to extract protein interface labels from pdb file[s]."""
import logging

import click
import numpy as np
import pandas as pd
import scipy.spatial as spa

import atom3d.util.datatypes as dt


index_columns = ['structure', 'model', 'chain', 'residue']


@click.command(help='Find neighbors for provided PDB files and output.')
@click.argument('input_pdbs', nargs=-1, type=click.Path(exists=True))
@click.argument('output_labels', type=click.Path())
@click.option('-b', '--bound_pdbs', multiple=True,
              type=click.Path(exists=True),
              help='If provided, use these PDB files to define the neighbors.')
@click.option('-c', '--cutoff', type=int, default=8,
              help='Maximum distance (in angstroms), for two residues to be '
              'considered neighbors.')
@click.option('--cutoff-type', default='CA',
              type=click.Choice(['heavy', 'CA'], case_sensitive=False),
              help='How to compute distance between residues: CA is based on '
              'alpha-carbons, heavy is based on any heavy atom.')
def get_neighbors_main(input_pdbs, output_labels, bound_pdbs, cutoff,
                       cutoff_type):
    input_dfs = [dt.bp_to_df(dt.read_pdb(x)) for x in input_pdbs]
    bound_dfs = [dt.bp_to_df(dt.read_pdb(x)) for x in bound_pdbs]

    neighbors, _ = get_neighbors(input_dfs, bound_dfs, cutoff, cutoff_type)
    # Write label file.
    neighbors.to_csv(output_labels, index=False)


def get_neighbors(subunits, cutoff, cutoff_type):
    """Given subunits, and optionally bound dfs, generate neighbors."""

    # Extract neighboring pairs of residues.  These are defined as pairs of
    # residues that are close to one another while spanning different subunits.
    neighbors = []
    used_pairs = []
    for i in range(len(subunits)):
        for j in range(i + 1, len(subunits)):
            if cutoff_type == 'CA':
                curr = _get_ca_neighbors(
                    subunits[i]['bound'], subunits[j]['bound'], cutoff)
            else:
                curr = _get_heavy_neighbors(
                    subunits[i]['bound'], subunits[j]['bound'], cutoff)
            neighbors.append(curr)
            if len(curr) > 0:
                used_pairs.append((subunits[i]['name'], subunits[j]['name']))

    if len(neighbors) > 0:
        neighbors = pd.concat(neighbors)
    else:
        neighbor_columns = [x + '0' for x in index_columns] + \
            [x + '1' for x in index_columns]
        neighbors = pd.DataFrame([], columns=neighbor_columns)
    neighbors['label'] = 1

    # Remove entries that are not present in input structures.
    _, res_to_idx = _get_idx_to_res_mapping(
        pd.concat([x['unbound'] for x in subunits]))
    to_drop = []
    for i, neighbor in neighbors.iterrows():
        res0 = tuple(neighbor[['structure0', 'model0', 'chain0', 'residue0']])
        res1 = tuple(neighbor[['structure1', 'model1', 'chain1', 'residue1']])
        if res0 not in res_to_idx or res1 not in res_to_idx:
            to_drop.append(i)
    logging.info(
        f'Removing {len(to_drop):} / {len(neighbors):} due to no matching '
        f'residue in unbound.')
    neighbors = neighbors.drop(to_drop).reset_index(drop=True)
    return neighbors, used_pairs


def get_negatives(subunits, neighbors):
    for i in range(len(subunits)):
        for j in range(i + 1, len(subunits)):
            negatives = _get_negatives(
                neighbors, subunits[i]['unbound'], subunits[j]['unbound'])
    return negatives


def get_res(df):
    """Get all residues."""
    return df[index_columns].drop_duplicates()


def get_subunits(input_dfs, bound_dfs):
    """Extract subunits to define protein interfaces for."""
    names = []

    if len(bound_dfs) > 0:
        # If bound pdbs are provided, we use their atoms to define which
        # residues are neighboring in the input files.
        # If we provide bound pdbs, we assume they are in a one-to-one
        # correspondence to the input pdbs, and that they define each
        # individual subunit we are trying to predict interfaces for.  We also
        # assume that model/chain/residue correspondence is exact between the
        # bound and input files.
        if len(bound_dfs) != len(input_dfs):
            raise RuntimeError('If providing bound dfs, provide same number '
                               'and in same order as input dfs.')

        # Mapping from bound name to unbound name.
        bound_subunits = []
        for b, i in zip(bound_dfs, input_dfs):
            i_name = i[['structure', 'model']].drop_duplicates()
            b_name = b[['structure', 'model']].drop_duplicates()
            if len(b_name) > 1 or len(i_name) > 1:
                raise RuntimeError('Multiple structure names in single df.')
            i_name = tuple(i_name.iloc[0])
            tmp = b.copy()
            tmp['structure'] = i_name[0]
            names.append(i_name)
            bound_subunits.append(tmp)

        unbound_subunits = input_dfs
    else:
        # If bound pdbs are not provided, we directly use the input files to
        # define which residues are neighboring.  In this case, we also assume
        # that each individual chain is its own subunit we are trying to
        # predict interfaces for.
        df = pd.concat(input_dfs)
        bound_subunits = []
        for name, x in df.groupby(['structure', 'model', 'chain']):
            names.append(name)
            bound_subunits.append(x)
        unbound_subunits = bound_subunits
    return [{'name': n, 'unbound': us, 'bound': bs}
            for (n, us, bs) in zip(names, unbound_subunits, bound_subunits)]


def _get_negatives(neighbors, df0, df1):
    """Get negative pairs, given positives."""
    idx_to_res0, res_to_idx0 = _get_idx_to_res_mapping(df0)
    idx_to_res1, res_to_idx1 = _get_idx_to_res_mapping(df1)
    all_pairs = np.zeros((len(idx_to_res0.index), len(idx_to_res1.index)))
    for i, neighbor in neighbors.iterrows():
        res0 = tuple(neighbor[['structure0', 'model0', 'chain0', 'residue0']])
        res1 = tuple(neighbor[['structure1', 'model1', 'chain1', 'residue1']])
        idx0 = res_to_idx0[res0]
        idx1 = res_to_idx1[res1]
        all_pairs[idx0, idx1] = 1
    pairs = np.array(np.where(all_pairs == 0)).T
    res0 = idx_to_res0.iloc[pairs[:, 0]][index_columns]
    res1 = idx_to_res1.iloc[pairs[:, 1]][index_columns]
    res0 = res0.reset_index(drop=True).add_suffix('0')
    res1 = res1.reset_index(drop=True).add_suffix('1')
    res = pd.concat((res0, res1), axis=1)
    return res


def _get_idx_to_res_mapping(df):
    """Define mapping from residue index to single id number."""
    idx_to_res = get_res(df).reset_index(drop=True)
    res_to_idx = idx_to_res.reset_index().set_index(index_columns)['index']
    return idx_to_res, res_to_idx


def _get_ca_neighbors(df0, df1, cutoff):
    """Get neighbors for alpha-carbon based distance."""
    ca0 = df0[df0['name'] == 'CA']
    ca1 = df1[df1['name'] == 'CA']

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


def lookup_subunit(name, df):
    """Lookup up subunit by name."""
    db5 = len(name) == 2
    if db5:
        bsel = (df['structure'] == name[0]) & (df['model'] == name[1])
        usel = bsel
    else:
        bsel = (df['structure'] == name[0]) & (df['model'] == name[1]) & \
            (df['chain'] == name[2])
        usel = (df['structure'] == name[0]) & (df['model'] == name[1] - 1) & \
            (df['chain'] == name[2])
    bound = df[bsel]
    unbound = df[usel]
    return {'name': name, 'bound': bound, 'unbound': unbound}



if __name__ == "__main__":
    get_neighbors_main()
