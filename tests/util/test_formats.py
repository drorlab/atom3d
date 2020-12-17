import pytest
import os
from pathlib import Path

import atom3d.util.formats as fo


def test_read_any_pdb():
    fo.read_any('tests/test_data/pdb/103l.pdb')
    fo.read_any('tests/test_data/pdb/117e.pdb')
    fo.read_any('tests/test_data/pdb/11as.pdb')
    fo.read_any('tests/test_data/pdb/2olx.pdb')

def test_read_any_pdb_gz():
    pass

def test_read_any_mmcif():
    fo.read_any('tests/test_data/mmcif/1j36_protein.cif')
    fo.read_any('tests/test_data/mmcif/1j36_protein.cif')
    fo.read_any('tests/test_data/mmcif/6h29_pocket.cif')
    fo.read_any('tests/test_data/mmcif/6h2t_protein.cif')

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



