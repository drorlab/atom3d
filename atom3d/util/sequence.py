"""Code relating to biomolecular sequence."""
import os
import subprocess

from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastpCommandline
import tqdm

import atom3d.util.datatypes as dt


def find_similar(chain_sequences, blast_db, cutoff, dataset_size):
    """Find all other pdbs that have sequence identity greater than cutoff."""
    sim = set()
    for chain, s in chain_sequences:
        blastp_cline = NcbiblastpCommandline(
            db=blast_db,
            outfmt="10 nident sacc",
            num_alignments=dataset_size,
            cmd='/usr/local/ncbi/blast/bin/blastp')
        out, err = blastp_cline(stdin=s)

        for res in out.split():
            nident, match = res.split(',')
            ref_pdb = match.split('_')[0]
            seq_id = float(nident) / len(s)
            if seq_id >= cutoff:
                sim.add(ref_pdb)

    return sim


####################################
# Sequence-level clustering code
####################################


def get_pdb_clusters(id_level, pdb_ids=None):
    """
    Downloads pre-calculated clusters from PDB at given cutoff.
    Returns dictionaries mapping PDB IDs to cluster IDs and vice versa.
    """
    id_level = int(id_level * 100)
    if id_level not in [30, 40, 50, 70, 90, 95, 100]:
        raise Exception(
            'invalid invalid identity cutoff. '
            'possible values = 30,40,50,70,90,95,100')
    print('getting clusters from PDB...')
    subprocess.call(
        'wget ftp://resources.rcsb.org/sequence/clusters/bc-{}.out'.format(
            id_level),
        shell=True)
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

    subprocess.check_output(
        'makeblastdb -in ' +
        blast_db +
        ' -dbtype prot',
        shell=True)


def get_chain_sequences(pdb_file):
    """
    Returns list of chain sequences from PDB file
    """
#     fname = os.path.join(path, pdb, pdb + '_protein.pdb')
    chain_seqs = []
    for seq in SeqIO.parse(pdb_file, 'pdb-atom'):
        try:
            pdb_id = seq.idcode
        except BaseException:
            pdb_id = os.path.basename(pdb_file).rstrip('.pdb')
        chain = pdb_id + '_' + seq.annotations['chain']
        chain_seqs.append((chain, str(seq.seq)))
    return chain_seqs


def get_all_chain_sequences(pdb_dataset):
    """Return list of tuples of (pdb_code, chain_sequences) for PDB dataset."""
    return [(dt.get_pdb_code(p), get_chain_sequences(p))
            for p in tqdm.tqdm(pdb_dataset)]


def write_fasta(chain_sequences, outfile):
    """
    Write chain_sequences to fasta file.
    """
    with open(outfile, 'w') as f:
        for chain, seq in chain_sequences.items():
            f.write('>' + chain + '\n')
            f.write(seq + '\n')
