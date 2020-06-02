"""Code relating to biomolecular sequence."""
import os
import subprocess

from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastpCommandline
import Bio.PDB.Polypeptide as poly
import dotenv as de
import numpy as np
import tqdm

import atom3d.util.datatypes as dt
import atom3d.util.log as log
import atom3d.util.file as fi


project_root = os.path.abspath(os.path.join(__file__, '../../..'))
de.load_dotenv(os.path.join(project_root, '.env'))
logger = log.getLogger('sequence')


def find_similar(chain_sequences, blast_db, cutoff, num_alignments):
    """Find all other pdbs that have sequence identity greater than cutoff."""

    if 'BLAST_BIN' not in os.environ:
        raise RuntimeError('Need to set BLAST_BIN in .env to use blastp')

    sim = set()
    for chain, s in chain_sequences:
        blastp_cline = NcbiblastpCommandline(
            db=blast_db,
            outfmt="10 nident sacc",
            num_alignments=num_alignments,
            cmd=os.path.join(os.environ['BLAST_BIN'], 'blastp'))
        out, err = blastp_cline(stdin=s)

        for res in out.split():
            nident, match = res.split(',')
            seq_id = float(nident) / len(s)
            if seq_id >= cutoff:
                sim.add(match)
    return list(sim)


####################################
# Sequence-level clustering code
####################################


def get_pdb_clusters(id_level, pdb_ids=None):
    """
    Downloads pre-calculated clusters from PDB at given cutoff.
    Returns dictionaries mapping PDB IDs to cluster IDs and vice versa.
    """
    id_level = int(id_level)
    if id_level not in [30, 40, 50, 70, 90, 95, 100]:
        raise Exception(
            'invalid invalid identity cutoff. '
            'possible values = 30,40,50,70,90,95,100')
    logger.info('getting clusters from PDB...')
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
                if pdb_ids is not None and not np.isin(pdb, pdb_ids).any():
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
    """Write provided chain sequences to blast db, for use with BLAST.

    Inputs:
    - all_chain_sequences: list of (structure name, chain_sequences)
    """
    logger.info(f'writing {len(all_chain_sequences):} chain sequences '
                f'to BLAST db {blast_db:}')

    if 'BLAST_BIN' not in os.environ:
        raise RuntimeError('Need to set BLAST_BIN in .env to use makeblastdb')

    flat_map = {}
    for (structure_name, cs) in all_chain_sequences:
        for (chain_name, seq) in cs:
            name = tuple_to_fasta_name(structure_name, chain_name)
            flat_map[name] = seq

    write_fasta(flat_map, blast_db)

    cmd = os.path.join(os.environ['BLAST_BIN'], 'makeblastdb')
    subprocess.check_output(f'{cmd:} -in {blast_db:} -dbtype prot', shell=True)


def tuple_to_fasta_name(structure_name, chain_name):
    """Fasta names can only be strings. This code writes the tuples in."""
    sname = '___'.join([str(x) for x in structure_name])
    cname = '___'.join([str(x) for x in chain_name])
    return sname + '____' + cname


def fasta_name_to_tuple(x):
    """Fasta names can only be strings. This code gets the tuples back out."""
    stuple = tuple(x.split('____')[0].split('___'))
    ctuple = tuple(x.split('____')[1].split('___'))
    stuple = (stuple[0], int(stuple[1]), stuple[2])
    return (stuple, ctuple)


def get_chain_sequences(pdb_file):
    """
    Return list of chain sequences from PDB file.

    Takes the form of list of (chain name, sequence string).
    """
#     fname = os.path.join(path, pdb, pdb + '_protein.pdb')
    chain_seqs = []
    for seq in SeqIO.parse(pdb_file, 'pdb-atom'):
        chain = seq.annotations['chain']
        chain_seqs.append(((chain,), str(seq.seq)))
    return chain_seqs


def get_all_chain_sequences(pdb_dataset):
    """Return list of tuples of (pdb_code, chain_sequences) for PDB dataset."""
    return [((fi.get_pdb_code(p),), get_chain_sequences(p))
            for p in tqdm.tqdm(pdb_dataset)]


def get_all_chain_sequences_df(df):
    """Return list of tuples of (struct_name, chain_sequences) for sharded."""
    all_chain_sequences = []
    # Keep only CA of standard residues
    df = df[df['name'] == 'CA'].drop_duplicates()
    df = df[df['resname'].apply(lambda x: poly.is_aa(x, standard=True))]
    df['resname'] = df['resname'].apply(poly.three_to_one)
    for s, structure in df.groupby(
            ['ensemble', 'subunit', 'structure']):
        chain_sequences = []
        for c, chain in structure.groupby(['model', 'chain']):
            seq = ''.join(chain['resname'])
            chain_sequences.append((c, seq))
        all_chain_sequences.append((s, chain_sequences))
    return all_chain_sequences


def write_fasta(chain_sequences, outfile):
    """
    Write chain_sequences to fasta file.
    """
    with open(outfile, 'w') as f:
        for chain, seq in chain_sequences.items():
            f.write('>' + chain + '\n')
            f.write(seq + '\n')
