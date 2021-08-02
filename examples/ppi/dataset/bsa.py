import click
import freesasa
import pandas as pd

import neighbors as nb
import atom3d.shard.shard as sh
import atom3d.util.formats as dt

freesasa.setVerbosity(freesasa.nowarnings)


@click.command(help='Find buried surface area (bsa) for entry in sharded.')
@click.argument('sharded_path', type=click.Path())
@click.argument('ensemble')
def compute_all_bsa_main(sharded_path, ensemble):
    sharded = sh.Sharded.load(sharded_path)
    ensemble = sharded.read_ensemble(ensemble)
    _, (bdf0, bdf1, udf0, udf1) = nb.get_subunits(ensemble)
    print(compute_bsa(bdf0, bdf1))


def compute_bsa(df0, df1, asa0=None, asa1=None):
    """Given two subunits, compute bsa."""
    result = {}
    bound = _merge_dfs(df0, df1)
    complex_asa = _compute_asa(bound)
    if asa0 is None:
        asa0 = _compute_asa(df0)
    if asa1 is None:
        asa1 = _compute_asa(df1)
    buried_surface_area = asa0 + asa1 - complex_asa
    result['bsa'] = buried_surface_area
    result['complex_asa'] = complex_asa
    result['asa0'] = asa0
    result['asa1'] = asa1
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
    structure = freesasa.Structure(
        classifier=freesasa.Classifier.getStandardClassifier('naccess'),
        options={'hydrogen': True, 'skip-unknown': True})
    for i, atom in df.iterrows():
        if atom['resname'] != 'UNK' and atom['element'] != 'H':
            structure.addAtom(
                atom['name'], atom['resname'], atom['residue'],
                atom['chain'], atom['x'], atom['y'], atom['z'])
    result = freesasa.calc(structure)
    return result.totalArea()


if __name__ == "__main__":
    compute_all_bsa_main()
