import pandas as pd
import atom3d.datasets as da
import atom3d.filters.filters as filters
import atom3d.filters.pdb as pdb


dataset = da.load_dataset('tests/test_data/lmdb', 'lmdb')

PDB_ENTRY_TYPE_FILE = 'atom3d/data/metadata/pdb_entry_type.txt'
RESOLUTION_FILE = 'atom3d/data/metadata/resolu.idx'


def test_size_filter():
    filter_fn = pdb.form_size_filter(max_size=800, min_size=80)
    for d in dataset:
        df_inp = d['atoms']
        df_fil = filter_fn(d['atoms'])
        # Determine manually whether the frame should be deleted
        num_res = len(df_inp[['model', 'chain', 'residue']].drop_duplicates())
        delete = num_res < 80 or num_res > 800
        # Check whether it has been deleted
        if delete: 
            assert len(df_fil) == 0
        else:
            assert len(df_fil) == len(df_inp)
        
        
def test_source_filter():
    # Test 'excluded' option
    filter_fn = pdb.form_source_filter(excluded=['diffraction'])
    for i,d in enumerate(dataset):
        df_fil = filter_fn(d['atoms'])
        assert len(df_fil) == 0
    # Test 'allowed' option
    filter_fn = pdb.form_source_filter(allowed=['diffraction'])
    for i,d in enumerate(dataset):
        df_fil = filter_fn(d['atoms'])
        assert len(df_fil) == len(d['atoms'])

        
def test_molecule_type_filter():
    # Test option that includes all examples (only proteins)
    filter_fn = pdb.form_molecule_type_filter(['prot'])
    for i,d in enumerate(dataset):
        df_fil = filter_fn(d['atoms'])
        assert len(df_fil) == len(d['atoms'])
    # Test option that excludes all examples (only proteins)
    filter_fn = pdb.form_molecule_type_filter(['nuc'])
    for i,d in enumerate(dataset):
        df_fil = filter_fn(d['atoms'])
        assert len(df_fil) == 0
        

def test_resolution_filter():   
    filter_fn = pdb.form_resolution_filter(threshold=2)
    reference = [0,0,105,1404]
    for i,d in enumerate(dataset):
        df_inp = d['atoms']
        df_fil = filter_fn(d['atoms'])
        assert len(df_fil) == reference[i]
        
