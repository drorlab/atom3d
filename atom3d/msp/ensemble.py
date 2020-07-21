"""Code to ensemble SKEMPI dataset."""
import os


def mut_ensembler(pdb_files):
    """We find matching original pdb for each mutated pdb."""
    dirs = list(set([os.path.dirname(f) for f in pdb_files]))

    original, mutated = {}, {}
    for f in pdb_files:
        name = os.path.splitext(os.path.basename(f))[0]
        if len(name.split('_')) > 3:
            assert name not in mutated
            mutated[name] = f
        else:
            assert name not in original
            original[name] = f

    ensembles = {}
    for mname, mfile in mutated.items():
        oname = '_'.join(mname.split('_')[:-1])
        ofile = original[oname]
        ensembles[mname] = {
            'original': ofile,
            'mutated': mfile
        }
    return ensembles
