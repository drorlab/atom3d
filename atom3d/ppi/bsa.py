import click
import freesasa
import pandas as pd

import atom3d.ppi.neighbors as nb
import atom3d.util.datatypes as dt

@click.command(help='Find buried surface area (bsa) for provided pdb files.')
@click.argument('input_pdbs', nargs=-1, type=click.Path(exists=True))
@click.option('-b', '--bound_pdbs', multiple=True,
              type=click.Path(exists=True),
              help='If provided, use these PDB files to define bound complex.')
def compute_all_bsa_main(input_pdbs, bound_pdbs):
    input_dfs = [dt.bp_to_df(dt.read_pdb(x)) for x in input_pdbs]
    bound_dfs = [dt.bp_to_df(dt.read_pdb(x)) for x in bound_pdbs]
    subunits = nb.get_subunits(input_dfs, bound_dfs)
    print(compute_all_bsa(subunits))


def compute_all_bsa(subunits):
    """Compute bsa between all possible subunit pairings."""
    results = []
    asas = [_compute_asa(subunit['unbound']) for subunit in subunits]

    for i in range(len(subunits)):
        for j in range(i + 1, len(subunits)):
            results.append(compute_bsa(
                subunits[i], subunits[j], asas[i], asas[j]))
    results = pd.concat(results, axis=1).T
    return results


def compute_bsa(subunit0, subunit1, asa0=None, asa1=None):
    """Given two subunits, compute bsa."""
    result = {}
    bound = _merge_dfs(subunit0['bound'], subunit1['bound'])
    complex_asa = _compute_asa(bound)
    if asa0 is None:
        asa0 = _compute_asa(subunit0['unbound'])
    if asa1 is None:
        asa1 = _compute_asa(subunit1['unbound'])
    buried_surface_area = asa0 + asa1 - complex_asa
    result['bsa'] = buried_surface_area
    result['complex_asa'] = complex_asa
    result['asa0'] = asa0
    result['asa1'] = asa1
    result['subunit0'] = subunit0['name']
    result['subunit1'] = subunit1['name']
    return pd.Series(result)


def _merge_dfs(df0, df1):
    """Create merged structure."""
    tmp0 = df0.copy()
    tmp1 = df1.copy()
    count = 0
    mapping0 = {v: i for i, v in enumerate(tmp0['model'].unique())}
    mapping1 = {v: (i + len(mapping0))
                for i, v in enumerate(tmp1['model'].unique())}
    tmp0['model'] = tmp0['model'].apply(lambda x: mapping0[x])
    tmp1['model'] = tmp1['model'].apply(lambda x: mapping1[x])
    result = pd.concat([tmp0, tmp1])
    result['structure'] = 'merged'
    return result


def _compute_asa(df):
    """Compute solvent-accessible surface area for provided strucutre."""
    bp = dt.df_to_bp(df)
    structure = freesasa.structureFromBioPDB(
        bp, options={'hydrogen': True, 'skip-unknown': True})
    result = freesasa.calc(structure)
    return result.totalArea()


if __name__ == "__main__":
    compute_all_bsa_main()
