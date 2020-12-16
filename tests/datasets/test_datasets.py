import pytest

import atom3d.datasets as da

def test_load_dataset():
    dataset = da.load_dataset('tests/test_data', 'pdb')
    assert len(dataset) == 4