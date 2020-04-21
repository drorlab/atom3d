import os
import sys
sys.path.append('..')
from util import datatypes as dt
from rdkit import Chem

def get_protein(protfile):
    prot = dt.read_pdb(protfile)
    return prot

def get_ligand(ligfile):
    lig = Chem.MolFromMol2File(ligfile)
    lig = Chem.RemoveHs(lig)
    return lig

def pairwise_identity(sequences):
    """
    Sequences is a list of sequences returned by get_sequence, so each item is a list of at least one sequence.
    Returns maximum ID between sets of sequences
    """
    id_mat = np.zeros((len(sequences), len(sequences)))
    for i in tqdm(range(len(sequences))):
        for j in range(len(sequences)):
            seqs1 = sequences[i]
            seqs2 = sequences[j]
            if len(seqs1) == 1 and len(seqs2) == 1:
                s1 = seqs1[0]
                s2 = seqs2[0]
                align = pairwise2.align.globalxx(s1, s2)
                score = align[0][2]
                length = min(len(s1), len(s2))
                identity = score / length
            else:
                ids = []
                for s1 in seqs1:
                    for s2 in seqs2:
                        align = pairwise2.align.globalxx(s1, s2)
                        score = align[0][2]
                        length = min(len(s1), len(s2))
                        identity = score / length
                        ids.append(identity)
                identity = np.max(ids)
            id_mat[i,j] = identity
    return id_mat

def get_pdb_clusters(id_level):
    os.system(f'wget ftp://resources.rcsb.org/sequence/clusters/bc-{id_level}.out')
    pdb2cluster = {}
    cluster2pdb = {}
    with open(f'bc-{id_level}.out') as f:
        for i, line in enumerate(f):
            cluster2pdb[i] = set()
            pdbs = line.strip().split()
            pdbs = [p[:4].lower() for p in pdbs]
            for pdb in pdbs:
                if pdb not in pdb2cluster:
                    pdb2cluster[pdb] = set()
                pdb2cluster[pdb].add(i)
                cluster2pdb[i].add(pdb)
            
    for pdb, clust in pdb2cluster.items():
        pdb2cluster[pdb] = list(clust)
    for clust, pdb in cluster2pdb.items():
        cluster2pdb[clust] = list(pdb)
    
    os.system(f'rm bc-{id_level}.out')
    
    return pdb2cluster, cluster2pdb