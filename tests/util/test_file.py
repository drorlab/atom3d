import pytest
import os
from pathlib import Path

import atom3d.util.file as fi


pdb_path = 'tests/test_data/pdb'
test_file_list = [pdb_path+'/103l.pdb',  
                  pdb_path+'/117e.pdb',   
                  pdb_path+'/11as.pdb',   
                  pdb_path+'/2olx.pdb']


def test_find_files():
    file_list = fi.find_files(pdb_path, 'pdb', relative=None)
    assert file_list == [Path(x) for x in test_file_list]


def test_get_pdb_code():
    codes = []
    for path in test_file_list:
        codes.append(fi.get_pdb_code(path))
    assert codes == ['103l','117e','11as','2olx']


def test_get_pdb_name():
    codes = []
    for path in test_file_list:
        codes.append(fi.get_pdb_name(path))
    assert codes == ['103l.pdb','117e.pdb','11as.pdb','2olx.pdb']

