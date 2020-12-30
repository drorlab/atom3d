import pytest
import os
import importlib
import torch
import numpy as np
import atom3d.datasets as da
import atom3d.splits.splits as spl
import atom3d.splits.sequence as seqspl


cutoff = 0.3
seq_dataset = da.load_dataset('tests/test_data/lmdb', 'lmdb')

@pytest.mark.network
def test_cluster_split():
    cutoff = 0.3
    # Perform the split
    s = seqspl.cluster_split(seq_dataset, cutoff, 
                             val_split=0.25, test_split=0.25, 
                             min_fam_in_split=1, random_seed=0)
    train_dataset, val_dataset, test_dataset = s
    # Compare split to what it should be (one sequence is thrown out)
    for ds in s: assert len(ds) == 1
    

@pytest.mark.skipif('BLAST_BIN' not in os.environ,
                    reason="Identity split requires BLAST!")
def test_identity_split():
    # Perform the split
    s = seqspl.identity_split(seq_dataset, cutoff, 
                              val_split=0.25, test_split=0.25,
                              min_fam_in_split=1, blast_db=None, 
                              random_seed=0)
    train_dataset, val_dataset, test_dataset = s
    # Compare split to what it should be
    ref_len = [2,1,1]
    for i,ds in enumerate(s): 
        assert len(ds) == ref_len[i]
    

