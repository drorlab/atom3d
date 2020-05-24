"""Ensembling operations for shards."""
import pandas as pd
import sys
sys.path.append('../..')

import atom3d.lap.ensemble as lape
import atom3d.ppi.db5 as db5
import atom3d.psp.casp as casp
import atom3d.util.datatypes as dt


def identity_ensembler(pdb_files):
    return {x: None for x in pdb_files}


# An ensembler maps a list of files to a 2-level dictionary.  First key is
# name of an ensemble, second key is name of a subunit.  A (ensemble, subunit)
# pair should map to a single file.
ensemblers = {
    'db5': db5.db5_ensembler,
    'casp': casp.casp_ensembler,
    'lap': lape.lap_ensembler,
    'none': identity_ensembler,
}


def parse_ensemble(name, ensemble):
    if ensemble is None:
        df = dt.bp_to_df(dt.read_any(name))
    else:
        df = []
        for subunit, f in ensemble.items():
            curr = dt.bp_to_df(dt.read_any(f))
            curr['subunit'] = subunit
            df.append(curr)
        df = pd.concat(df)
        df['ensemble'] = name
    return df
