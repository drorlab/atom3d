import pytest
import importlib

import atom3d.util.formats as fo


# Tests for reading PDB-like formats

nrres = {'103l':290,'117e':1029,'11as':740,'2olx':4}

def test_read_any_pdb():
    for c in nrres.keys():
        bp = fo.read_any('tests/test_data/pdb/'+c+'.pdb')
        nr = len([r for r in bp.get_residues()])
        assert nr==nrres[c]
        
def test_read_any_pdb_gz():
    for c in nrres.keys():
        bp = fo.read_any('tests/test_data/pdbgz/'+c+'.pdb.gz')
        nr = len([r for r in bp.get_residues()])
        assert nr==nrres[c]
        
def test_read_any_mmcif():
    for c in nrres.keys():
        bp = fo.read_any('tests/test_data/pdb/'+c+'.pdb')
        nr = len([r for r in bp.get_residues()])
        assert nr==nrres[c]


# Tests for reading small-molecule formats

@pytest.mark.skipif(not importlib.util.find_spec("rdkit") is not None,
                    reason="Reading SDF files requires RDKit!")
def test_read_any_sdf():
    fo.read_any('tests/test_data/sdf/1j01_ligand.sdf')
    fo.read_any('tests/test_data/sdf/2yme_ligand.sdf')  
    fo.read_any('tests/test_data/sdf/4tjz_ligand.sdf')  
    fo.read_any('tests/test_data/sdf/6b4n_ligand.sdf')

def test_read_any_xyz():
    pass

def test_read_xyz_gdb():
    fo.read_xyz('tests/test_data/xyz-gdb/dsgdb9nsd_000005.xyz', gdb=True)
    fo.read_xyz('tests/test_data/xyz-gdb/dsgdb9nsd_000212.xyz', gdb=True)  
    fo.read_xyz('tests/test_data/xyz-gdb/dsgdb9nsd_001458.xyz', gdb=True)



