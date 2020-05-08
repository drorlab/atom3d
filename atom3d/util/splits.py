import numpy as np
import subprocess
import tqdm
import os, sys
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio import SeqIO
sys.path.append('../..')

import atom3d.util.datatypes as dt


# Splits data into test, validation, and training sets.

def random_split(dataset_size,train_split=None,vali_split=0.1,test_split=0.1,shuffle=True,random_seed=None,exclude=None):
    """Creates data indices for training and validation splits.

        Args:
            dataset_size (int): number of elements in the dataset
            vali_split (float): fraction of data used for validation. Default: 0.1
            test_split (float): fraction of data used for testing. Default: 0.1
            shuffle (bool):     indices are shuffled. Default: True
            random_seed (int):  specifies random seed for shuffling. Default: None

        Returns:
            indices_test (int[]):  indices of the test set.
            indices_vali (int[]):  indices of the validation set.
            indices_train (int[]): indices of the training set.

    """

    # Initialize the indices
    all_indices = np.arange(dataset_size,dtype=int)
    print('Splitting dataset with',len(all_indices),'entries.')

    # Delete all indices that shall be excluded
    if exclude is None:
        indices = all_indices
    else:
        print('Excluding',len(exclude),'entries.')
        to_keep = np.invert(np.isin(all_indices, exclude))
        indices = all_indices[to_keep]
        print('Remaining',len(indices),'entries.')
    num_indices = len(indices)

    # Calculate the numbers of elements per split
    vsplit = int(np.floor(vali_split * num_indices))
    tsplit = int(np.floor(test_split * num_indices))
    if train_split is not None:
        train = int(np.floor(train_split * num_indices))
    else:
        train = num_indices-vsplit-tsplit

    # Shuffle the dataset if desired
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Determine the indices of each split
    indices_test  = indices[:tsplit]
    indices_vali  = indices[tsplit:tsplit+vsplit]
    indices_train = indices[tsplit+vsplit:tsplit+vsplit+train]

    return indices_test, indices_vali, indices_train


def time_split(data, val_years, test_years):
    """
    Splits data into train, val, test by year.

    Args:
        data (DataFrame): year data, with columns named 'pdb' and 'year'
        val_years (str[]): years to include in validation set
        test_years (str[]): years to include in test set

    Returns:
        train_set (str[]):  pdbs in the train set
        val_set (str[]):  pdbs in the validation set
        test_set (str[]): pdbs in the test set
    """
    val = data[data.year.isin(val_years)]
    test = data[data.year.isin(test_years)]
    train = data[~data.pdb.isin(val.pdb.tolist() + test.pdb.tolist())]

    train_set = train['pdb'].tolist()
    val_set = val['pdb'].tolist()
    test_set = test['pdb'].tolist()

    return train_set, val_set, test_set


def read_split_file(split_file):
    """
    Reads text file with pre-defined split, one example per row, returning list of examples
    """
    with open(split_file) as f:
        # file may contain integer indices or string identifiers (e.g. PDB codes)
        lines = f.readlines()
        try:
            split = [int(x.strip()) for x in lines]
        except ValueError:
            split = [x.strip() for x in lines]
    return split



####################################
# split by pre-clustered sequence
# identity clusters from PDB
####################################

def cluster_split(all_chain_sequences, cutoff, val_split=0.1, test_split=0.1, min_fam_in_split=5, random_seed=None):
    """
    Splits pdb dataset into train, validation, and test using pre-computed sequence identity clusters from PDB

    Args:
        all_chain_sequences ((str, chain_sequences)[]): tuple of pdb ids and chain_sequences in dataset
        cutoff (float): sequence identity cutoff (can be .3, .4, .5, .7, .9, .95, 1.0)
        val_split (float): fraction of data used for validation. Default: 0.1
        test_split (float): fraction of data used for testing. Default: 0.1
        min_fam_in_split (int): controls variety of val/test sets. Default: 5
        random_seed (int):  specifies random seed for shuffling. Default: None

    Returns:
        train_set (str[]):  pdbs in the train set
        val_set (str[]):  pdbs in the validation set
        test_set (str[]): pdbs in the test set

    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(all_chain_sequences)
    test_size = n * test_split
    val_size = n * val_split
    max_hit_size_test = test_size / min_fam_in_split
    max_hit_size_val = val_size / min_fam_in_split

    np.random.shuffle(all_chain_sequences)
    pdb_ids = [p for (p, _) in all_chain_sequences]

    clusterings = get_pdb_clusters(cutoff, pdb_ids)

    print('generating validation set...')
    val_set, all_chain_sequences = create_cluster_split(
        all_chain_sequences, clusterings, cutoff, val_size, min_fam_in_split)
    print('generating test set...')
    test_set, all_chain_sequences = create_cluster_split(
        all_chain_sequences, clusterings, cutoff, test_size, min_fam_in_split)
    train_set = set([p for (p, _) in all_chain_sequences])

    print('train size', len(train_set))
    print('val size', len(val_set))
    print('test size', len(test_set))

    return train_set, val_set, test_set


def get_pdb_clusters(id_level, pdb_ids=None):
    """
    Downloads pre-calculated clusters from PDB at given cutoff.
    Returns dictionaries mapping PDB IDs to cluster IDs and vice versa.
    """
    id_level = int(id_level * 100)
    if id_level not in [30,40,50,70,90,95,100]:
        raise Exception('invalid invalid identity cutoff. possible values = 30,40,50,70,90,95,100')
    print('getting clusters from PDB...')
    subprocess.call('wget ftp://resources.rcsb.org/sequence/clusters/bc-{}.out'.format(id_level), shell=True)
    pdb2cluster = {}
    cluster2pdb = {}
    with open(f'bc-{id_level}.out') as f:
        for i, line in enumerate(f):
            cluster2pdb[i] = set()
            pdbs = line.strip().split()
            pdbs = [p[:4].lower() for p in pdbs]
            for pdb in pdbs:
                if pdb_ids is not None and pdb not in pdb_ids:
                    continue
                if pdb not in pdb2cluster:
                    pdb2cluster[pdb] = set()
                pdb2cluster[pdb].add(i)
                cluster2pdb[i].add(pdb)

    os.system(f'rm bc-{id_level}.out')

    return pdb2cluster, cluster2pdb


def find_cluster_members(pdb, clusterings):
    """
    Find all other pdbs that co-occur in a cluster with a given pdb
    """
    pdb2cluster, cluster2pdb = clusterings
    clusters = pdb2cluster[pdb]
    all_pdbs = set()
    for clust in clusters:
        pdbs = cluster2pdb[clust]
        all_pdbs = all_pdbs.union(pdbs)
    return all_pdbs


def create_cluster_split(all_chain_sequences, clusterings, cutoff, split_size,
                         min_fam_in_split):
    """
    Create a split while retaining diversity specified by min_fam_in_split.
    Returns split and removes any pdbs in this split from the remaining dataset
    """
    pdb_ids = [p for (p, _) in all_chain_sequences]
    split = set()
    idx = 0
    while len(split) < split_size:
        (rand_id, rand_cs) = all_chain_sequences[idx]
        split.add(rand_id)
        hits = find_cluster_members(rand_id, clusterings)
        # ensure that at least min_fam_in_split families in each split
        if len(hits) > split_size / min_fam_in_split:
            idx += 1
            continue
        split = split.union(hits)
        idx += 1

    for hit in split:
        loc = pdb_ids.index(hit)
        all_chain_sequences.pop(loc)
        pdb_ids.pop(loc)

    return split, all_chain_sequences


####################################
# split by calculating sequence identity
# to any example in training set
####################################

def identity_split(all_chain_sequences, cutoff, val_split=0.1, test_split=0.1, min_fam_in_split=5, blast_db=None, random_seed=None):
    """
    Splits pdb dataset into train, validation, and test using pre-computed sequence identity clusters from PDB

    Args:
        all_chain_sequences ((str, chain_sequences)[]): tuple of pdb ids and chain_sequences in dataset
        cutoff (float): sequence identity cutoff (can be .3, .4, .5, .7, .9, .95, 1.0)
        val_split (float): fraction of data used for validation. Default: 0.1
        test_split (float): fraction of data used for testing. Default: 0.1
        min_fam_in_split (int): controls variety of val/test sets. Default: 5
        blast_db (str): location of pre-computed BLAST DB for dataset. If None, compute and save in 'blast_db'. Default: None
        random_seed (int):  specifies random seed for shuffling. Default: None

    Returns:
        train_set (str[]):  pdbs in the train set
        val_set (str[]):  pdbs in the validation set
        test_set (str[]): pdbs in the test set

    """
    if blast_db is None:
        write_to_blast_db(all_chain_sequences, 'blast_db')
        blast_db = 'blast_db'

    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(all_chain_sequences)
    test_size = n * test_split
    val_size = n * val_split
    max_hit_size_test = test_size / min_fam_in_split
    max_hit_size_val = val_size / min_fam_in_split

    np.random.shuffle(all_chain_sequences)

    print('generating validation set...')
    val_set, all_chain_sequences = create_identity_split(
        all_chain_sequences, cutoff, val_size, min_fam_in_split)
    print('generating test set...')
    test_set, all_chain_sequences = create_identity_split(
        all_chain_sequences, cutoff, test_size, min_fam_in_split)
    train_set = set([p for (p, _) in all_chain_sequences])

    print('train size', len(train_set))
    print('val size', len(val_set))
    print('test size', len(test_set))

    return train_set, val_set, test_set


def find_similar(chain_sequences, blast_db, cutoff, dataset_size):
    """
    Find all other pdbs that have sequence identity greater than cutoff.
    """
    sim = set()
    for chain, seq in chain_sequences:
        blastp_cline = NcbiblastpCommandline(db=blast_db, outfmt="10 nident sacc", num_alignments=dataset_size, cmd='/usr/local/ncbi/blast/bin/blastp')
        out, err = blastp_cline(stdin=seq)

        for res in out.split():
            nident, match = res.split(',')
            ref_pdb = match.split('_')[0]
            seq_id = float(nident) / len(seq)
            if seq_id >= cutoff:
                sim.add(ref_pdb)

    return sim


def create_identity_split(all_chain_sequences, cutoff, split_size,
                          min_fam_in_split):
    """
    Create a split while retaining diversity specified by min_fam_in_split.
    Returns split and removes any pdbs in this split from the remaining dataset
    """
    dataset_size = len(all_chain_sequences)
    pdb_ids = [p for (p, _) in all_chain_sequences]
    split = set()
    idx = 0
    while len(split) < split_size:
        (rand_id, rand_cs) = all_chain_sequences[idx]
        split.add(rand_id)
        hits = find_similar(rand_cs, 'blast_db', cutoff, dataset_size)
        # ensure that at least min_fam_in_split families in each split
        if len(hits) > split_size / min_fam_in_split:
            idx += 1
            continue
        split = split.union(hits)
        idx += 1

    for hit in split:
        loc = pdb_ids.index(hit)
        all_chain_sequences.pop(loc)
        pdb_ids.pop(loc)

    return split, all_chain_sequences


####################################
# useful functions
####################################

def write_to_blast_db(all_chain_sequences, blast_db):
    """Write provided pdb dataset to blast db, for use with BLAST.

    Inputs:
    - all_chain_sequences: list of (pdb_id, chain_sequences)
    """
    print('writing chain sequences to BLAST db')
    flat_map = {}
    for cs in all_chain_sequences:
        for (chain, seq) in cs:
            flat_map[chain] = seq

    write_fasta(flat_map, blast_db)

    subprocess.check_output('makeblastdb -in ' + blast_db + ' -dbtype prot', shell=True)


def get_chain_sequences(pdb_file):
    """
    Returns list of chain sequences from PDB file
    """
#     fname = os.path.join(path, pdb, pdb + '_protein.pdb')
    chain_seqs = []
    for seq in SeqIO.parse(pdb_file, 'pdb-atom'):
        try:
            pdb_id = seq.idcode
        except:
            pdb_id = os.path.basename(pdb_file).rstrip('.pdb')
        chain = pdb_id + '_' + seq.annotations['chain']
        chain_seqs.append((chain,str(seq.seq)))
    return chain_seqs


def get_all_chain_sequences(pdb_dataset):
    """Return list of tuples of (pdb_code, chain_sequences) for PDB dataset."""
    return [(dt.get_pdb_code(p), get_chain_sequences(p))
            for p in tqdm.tqdm(pdb_dataset)]


def write_fasta(seq_dict, outfile):
    """
    Takes dictionary of sequences keyed by id and writes to fasta file
    """
    with open(outfile, 'w') as f:
        for i, s in seq_dict.items():
            f.write('>'+i+'\n')
            f.write(s + '\n')
