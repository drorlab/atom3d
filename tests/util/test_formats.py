import pytest
import importlib

import atom3d.util.formats as fo


# -- Reading PDB-like formats --

numres = {'103l':290,'117e':1029,'11as':740,'2olx':4}

def test_read_any_pdb():
    for c in numres.keys():
        bp = fo.read_any('tests/test_data/pdb/'+c+'.pdb')
        nr = len([r for r in bp.get_residues()])
        assert nr==numres[c]
        
def test_read_any_pdb_gz():
    for c in numres.keys():
        bp = fo.read_any('tests/test_data/pdbgz/'+c+'.pdb.gz')
        nr = len([r for r in bp.get_residues()])
        assert nr==numres[c]
        
def test_read_any_mmcif():
    for c in numres.keys():
        bp = fo.read_any('tests/test_data/mmcif/'+c+'.cif')
        nr = len([r for r in bp.get_residues()])
        assert nr==numres[c]


# -- Reading SDF format --

numat_sdf = {'1j01':18,'2yme':23,'4tjz':12,'6b4n':46}

@pytest.mark.skipif(not importlib.util.find_spec("rdkit") is not None,
                    reason="Reading SDF files requires RDKit!")
def test_read_any_sdf():
    for c in numat_sdf.keys():
        bp = fo.read_any('tests/test_data/sdf/'+c+'_ligand.sdf')
        nr = len([a for a in bp.get_atoms()])
        assert nr==numat_sdf[c]


# -- Reading xyz and derived formats --

numat_gdb = {'000005':3,'000212':13,'001458':13}
inchi_gdb = {'000005':'InChI=1S/CHN/c1-2/h1H',
             '000212':'InChI=1S/C5H7N/c1-6-4-2-3-5-6/h2-5H,1H3',
             '001458':'InChI=1S/C3H6N2O2/c4-3(1-6)5-2-7/h2,6H,1H2,(H2,4,5,7)'}

# Standard xyz format
def test_read_any_xyz():
    for c in numat_gdb.keys():
        bp = fo.read_any('tests/test_data/xyz/dsgdb9nsd_'+c+'_pos.xyz')
        numat = len([a for a in bp.get_atoms()])
        assert numat_gdb[c] == numat

# The xyz format used in GDB dataset
# Modifications wrt standard:
# * data in the title line (2nd line)
# * an additional column with charges
# * a line with frequencies after the atom list
# * two lines with smiles and inchi
def test_read_xyz_gdb():
    for c in numat_gdb.keys():
        _ = fo.read_xyz('tests/test_data/xyz-gdb/dsgdb9nsd_'+c+'.xyz', gdb=True)
        bp, data, freq, smiles, inchi = _
        numat = len([a for a in bp.get_atoms()])
        assert numat_gdb[c] == numat
        assert inchi_gdb[c] == inchi


# -- Handling data frames --
#
#  These tests rely on the pdb reader
#

def test_merge_dfs():
    df_list = []
    num_at = []
    for c in numres.keys():
        bp = fo.read_any('tests/test_data/pdb/'+c+'.pdb')
        df_part = fo.bp_to_df(bp)
        num_at.append(len(df_part))
        df_list.append(df_part)
    df = fo.merge_dfs(df_list)
    assert len(df) == sum(num_at) == 11602

def test_merge_and_split_dfs():
    df_list = []
    num_at = []
    for c in numres.keys():
        bp = fo.read_any('tests/test_data/pdb/'+c+'.pdb')
        df_part = fo.bp_to_df(bp)
        num_at.append(len(df_part))
        df_list.append(df_part)
    df = fo.merge_dfs(df_list)
    assert len(df) == sum(num_at) == 11602
    df_split = fo.split_df(df, key="ensemble")
    num_at_split = [len(d[1]) for d in df_split] 
    assert num_at_split == num_at == [1404, 4943, 5220, 35]





