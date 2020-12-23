import numpy as np
import pandas as pd

import atom3d.datasets as da
import atom3d.protein.sequence as seq


def test_find_similar():
    # TODO: write test for seq.find_similar()
    pass

def test_get_pdb_clusters():
    id_level = 0.3
    pdb2cluster, cluster2pdb = seq.get_pdb_clusters(id_level, pdb_ids=None)
    for cluster in cluster2pdb.keys():
        for pdb in cluster2pdb[cluster]:
            assert cluster in pdb2cluster[pdb]
            
def test_find_cluster_members():
    id_level = 0.3
    clusterings = seq.get_pdb_clusters(id_level, pdb_ids=None)
    for pdb in clusterings[0].keys():
        seq.find_cluster_members(pdb, clusterings)

def test_get_chain_sequences():
    dataset = da.load_dataset('tests/test_data/lmdb', 'lmdb')
    cseq = seq.get_chain_sequences(dataset[2]['atoms'])
    assert cseq[0][1] == 'NNQQ'
    
def test_write_to_blast_db():
    # TODO: write test for seq.write_to_blast_db()
    pass
