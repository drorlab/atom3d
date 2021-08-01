"""Ensembling operations."""
import pandas as pd

import atom3d.datasets.ppi.db5 as db5
import atom3d.util.formats as dt


def identity_ensembler(pdb_files):
    return {x: None for x in pdb_files}


# An ensembler maps a list of files to an ensemble map (a 2-level dictionary).
#
# An ensemble map takes the form of a 2-level dictionary:
# (e -> (s -> sel))
#
# Where 'e' is the string name of an ensemble, and 's' is the string name of a
# subunit of the ensemble.  'sel' can be one of 1) a filename or 2) a pandas
# DataFrame, and corresponds to the subunit specified by (e, s).
ensemblers = {
    'db5': db5.db5_ensembler,
    'none': identity_ensembler,
}


def parse_ensemble(name, ensemble):
    if ensemble is None:
        df = dt.bp_to_df(dt.read_any(name))
    else:
        df = []
        for subunit, f in ensemble.items():
            if isinstance(f, pd.DataFrame):
                curr = f
            else:
                curr = dt.bp_to_df(dt.read_any(f))

            curr['subunit'] = subunit
            df.append(curr)
        df = pd.concat(df)
        df['ensemble'] = name
    return df
