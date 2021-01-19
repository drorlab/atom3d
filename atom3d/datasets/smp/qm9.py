"""QM9 functions."""
import collections as col
from atom3d.util.formats import *

def qm9_ensembler(sdf_files):
    
    bp = read_sdf(sdf_files)
    df = bp_to_df(bp)
    
    df['subunit'] = df['model']
    df['structure'] = df['model']
    df['ensemble'] = df['model']

    ensembles = {}
    subunits = {}
    for mol in df.groupby(['ensemble','model'],as_index=False):
        subunits[mol[0][1]] = mol[1]
    ensembles[None] = subunits

    return ensembles
    
