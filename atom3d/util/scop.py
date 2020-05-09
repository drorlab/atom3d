"""Functions related to SCOP classes of structures."""
import pandas as pd


SCOP_CLA_LATEST_FILE = 'metadata/scop-cla-latest.txt'


def get_scop_index():
    """Get index mapping from PDB code to SCOP classification."""

    scop = pd.read_csv(SCOP_CLA_LATEST_FILE, skiprows=6, delimiter=' ',
                       names=['pdb_code', 'scop'], usecols=[1, 10])
    scop['pdb_code'] = scop['pdb_code'].apply(lambda x: x.lower())
    scop['type'] = \
        scop['scop'].apply(lambda x: int(x.split(',')[0].split('=')[1]))
    scop['class'] = \
        scop['scop'].apply(lambda x: int(x.split(',')[1].split('=')[1]))
    scop['fold'] =  \
        scop['scop'].apply(lambda x: int(x.split(',')[2].split('=')[1]))
    scop['superfamily'] = \
        scop['scop'].apply(lambda x: int(x.split(',')[3].split('=')[1]))
    scop['family'] = \
        scop['scop'].apply(lambda x: int(x.split(',')[4].split('=')[1]))
    del scop['scop']
    scop = scop.set_index('pdb_code')
    return scop
