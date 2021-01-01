import numpy as np
import pandas as pd
import atom3d.datasets as da
import atom3d.filters.filters as filters


dataset = da.load_dataset('tests/test_data/lmdb', 'lmdb')

standard_amino_acids = ['THR', 'TYR', 'ARG', 'GLN', 'ILE', 
                        'GLY', 'ALA', 'LYS', 'ASN', 'LEU', 
                        'GLU', 'VAL', 'ASP', 'PRO', 'SER', 
                        'PHE', 'HIS', 'MET', 'TRP', 'CYS']


def test_standard_residue_filter():
    """Filter out non-standard residues."""
    ref_size = [5136, 4466, 105, 1270]
    for i, d in enumerate(dataset):
        df_fil = filters.standard_residue_filter(d['atoms'])
        # The remaining residue names have to be a subset of the standard AAs
        resnames = df_fil.resname.unique()
        assert set(resnames).issubset(set(standard_amino_acids))
        # Comparison to reference size
        assert len(df_fil) == ref_size[i]

        
def test_first_model_filter():
    """Remove anything beyond first model in structure."""
    ref_size = [5220, 4943, 35, 1404]
    for i, d in enumerate(dataset):
        df_fil = filters.first_model_filter(d['atoms'])
        # There's only one model in the filtered dataframe.
        assert len(df_fil.model.unique()) == 1
        # This model has to be the first one.
        assert df_fil.model.unique()[0] == d['atoms'].model.unique()[0]
        # Comparison to reference size
        assert len(df_fil) == ref_size[i]

        
def test_single_chain_filter():
    """Remove anything that has more than one model/chain."""
    for i, d in enumerate(dataset):
        df_inp = d['atoms']
        df_fil = filters.single_chain_filter(d['atoms'])
        # Determine manually whether the frame should be deleted
        delete = len(df_inp.model.unique()) > 1 or len(df_inp.chain.unique()) > 1
        # Check whether it has been deleted
        if delete: 
            assert len(df_fil) == 0
        else:
            assert len(df_fil) == len(df_inp)


def test_distance_filter():
    """Remove all atoms within a certain distance around a certain position"""
    for pos in [np.array([0,0,0]),np.array([[0,0,0],[0,0,0]])]:
        dist = 10.
        for i, d in enumerate(dataset):
            df_inp = d['atoms']
            df_fil = filters.distance_filter(d['atoms'], pos, dist)
            abscrd = np.array(df_inp['x'])**2 + np.array(df_inp['y'])**2 + np.array(df_inp['z'])**2
            assert len(df_fil) == np.sum(abscrd<dist**2)


def test_identity_filter():
    for d in dataset:
        df_fil = filters.identity_filter(d['atoms'])
        assert df_fil.equals(d['atoms']) 
