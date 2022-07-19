"""Functions related to SCOP classes of structures."""
import os

import pandas as pd


SCOP_CLA_LATEST_FILE = 'atom3d/data/metadata/scop-cla-latest.txt'
PDB_CHAIN_SCOP2_UNIPROT_FILE = 'atom3d/data/metadata/pdb_chain_scop2_uniprot.csv'
PDB_CHAIN_SCOP2B_SF_UNIPROT_FILE = 'atom3d/data/metadata/pdb_chain_scop2b_sf_uniprot.csv'


def get_scop_index():
    """Get index mapping from PDB code and chain to SCOP classification."""

    # Load core SCOP database.  Mapping from domains to classification.
    scop = pd.read_csv(
        SCOP_CLA_LATEST_FILE, skiprows=6, delimiter=' ',
        names=['fa-domid', 'fa-pdbid', 'fa-pdbreg', 'fa-uniid', 'fa-unireg',
               'sf-domid', 'sf-pdbid', 'sf-pdbreg', 'sf-uniid', 'sf-unireg',
               'scop'])
    scop['pdb_code'] = scop['fa-pdbid'].apply(lambda x: x.lower())
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

    # Load mapping of representatives to scop domains.
    scop2_uniprot = pd.read_csv(
        PDB_CHAIN_SCOP2_UNIPROT_FILE, skiprows=2,
        names=['pdb_code', 'chain', 'sp-primary', 'sf-domid', 'fa-domid'],
        error_bad_lines=False,
    )
    # Some superfamily entries are a bit messed up here, so we remove.
    scop2_uniprot = scop2_uniprot[
        scop2_uniprot['sf-domid'].astype(str).str.isnumeric()]
    scop2_uniprot['sf-domid'] = pd.to_numeric(scop2_uniprot['sf-domid'])

    # Load mapping of non-representatives to scop domains.
    scop2b_sf_uniprot = pd.read_csv(
        PDB_CHAIN_SCOP2B_SF_UNIPROT_FILE, skiprows=2, usecols=[0, 1, 2, 3],
        names=['pdb_code', 'chain', 'sf-domid', 'sp-primary'])

    # Now merge the mappings:
    # (pdb_code, chain) -> domain
    # AND
    # domain -> scop class
    # TO GET
    # (pdb_code, chain) -> scop class
    scop = scop.set_index('sf-domid')
    scop2_uniprot = scop2_uniprot.set_index('sf-domid')
    scop2b_sf_uniprot = scop2b_sf_uniprot.set_index('sf-domid')
    sf_to_scop = scop[['class', 'fold', 'superfamily', 'family']]
    sf_to_chain = pd.concat([
        scop2_uniprot[['pdb_code', 'chain']],
        scop2b_sf_uniprot[['pdb_code', 'chain']]])
    chain_to_scop = pd.merge(sf_to_scop, sf_to_chain, on='sf-domid')
    chain_to_scop = chain_to_scop.drop_duplicates().set_index(
        ['pdb_code', 'chain']).sort_index(level=[0, 1])
    return chain_to_scop
