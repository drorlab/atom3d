import collections as col

import Bio.PDB
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
    print(compute_all_bsa(input_dfs, bound_dfs))


def compute_all_bsa(input_dfs, bound_dfs):
    """Given input dfs, and optionally bound dfs, compute bsa."""
    names, unbound_subunits, bound_subunits = nb.subunits(input_dfs, bound_dfs)
    results = col.defaultdict(list)

    for i in range(len(bound_subunits)):
        for j in range(i + 1, len(bound_subunits)):
            bp0 = dt.df_to_bp(bound_subunits[i])
            bp1 = dt.df_to_bp(bound_subunits[j])
            bound = dt.bp_to_df(_merge_bps(bp0, bp1))
            (bsa, complex_asa, asa0, asa1) = compute_bsa(
                unbound_subunits[i], unbound_subunits[j], bound)
            results['bsa'].append(bsa)
            results['complex_asa'].append(complex_asa)
            results['asa0'].append(asa0)
            results['asa1'].append(asa1)
            results['subunit0'].append(names[i])
            results['subunit1'].append(names[j])
    results = pd.DataFrame(results)
    return results


def _merge_bps(bp0, bp1):
    """Create merged biopython structure."""
    combined_bp = Bio.PDB.Structure.Structure('merged')
    count = 0
    for model in bp0:
        model.id = count
        model.serial_num = count
        count += 1
        combined_bp.add(model)
    for model in bp1:
        model.id = count
        model.serial_num = count
        count += 1
        combined_bp.add(model)
    return combined_bp


def compute_bsa(df0, df1, bound_df=None):
    """
    Compute buried surface area across two structures.

    Can optionally provide bound complex too.  Otherwise, it is computed by
    just merging the two provided structure.
    """

    bp0 = dt.df_to_bp(df0)
    bp1 = dt.df_to_bp(df1)

    if bound_df is None:
        bound_bp = _merge_bps(bp0, bp1)
    else:
        bound_bp = dt.df_to_bp(bound_df)

    result, sasa_classes = freesasa.calcBioPDB(bound_bp)
    complex_asa = result.totalArea()

    result, sasa_classes = freesasa.calcBioPDB(bp0)
    asa0 = result.totalArea()

    result, sasa_classes = freesasa.calcBioPDB(bp1)
    asa1 = result.totalArea()

    buried_surface_area = asa0 + asa1 - complex_asa
    return (buried_surface_area, complex_asa, asa0, asa1)


if __name__ == "__main__":
    compute_all_bsa_main()
