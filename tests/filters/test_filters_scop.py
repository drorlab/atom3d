import pandas as pd
import atom3d.datasets as da
import atom3d.filters.filters as filters
import atom3d.filters.scop as scop


dataset = da.load_dataset('tests/test_data/lmdb', 'lmdb')


# TODO: Commented code below fails.
# FileNotFoundError: [Errno 2] File ../../metadata/scop-cla-latest.txt does not exist: '../../metadata/scop-cla-latest.txt'
def test_scop_filter():   
#    level = 'class'
#    filter_fn = scop.form_scop_filter(level, allowed=None, excluded=None)
#    for i,d in enumerate(dataset):
#        df_fil = filter_fn(d['atoms'])
    pass
        
        