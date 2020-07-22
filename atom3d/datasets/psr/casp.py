"""CASP functions."""
import collections as col

import atom3d.datasets.psr.util as util


def casp_ensembler(pdb_files):
    targets = col.defaultdict(list)
    for f in pdb_files:
        target_name = util.get_target_name(f)
        targets[target_name].append(f)

    # target_name -> (decoy_name -> filename)
    ensembles = {}
    for target_name, files in targets.items():
        subunits = {util.get_decoy_name(f): f for f in files}
        ensembles[target_name] = subunits

    return ensembles
