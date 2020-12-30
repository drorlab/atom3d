"""Code relating to biomolecular sequence."""
import os
import subprocess

import Bio.PDB.Polypeptide as Poly
import dotenv as de
import numpy as np
from Bio.Blast.Applications import NcbiblastpCommandline

import atom3d.util.log as log

project_root = os.path.abspath(os.path.join(__file__, '../../..'))
de.load_dotenv(os.path.join(project_root, '.env'))
logger = log.get_logger('sequence')


def find_similar(chain_sequences, blast_db, cutoff, num_alignments):
    """Find all other proteins that have sequence identity greater than ``cutoff`` to query protein given by ``chain_sequences``. Assumes sequences have already been processed into a BLAST-formatted database by :func:`write_to_blast_db`.

    :param chain_sequences: Query protein represented as a list of tuples of (id, sequence) for each chain, as output by :func:`get_chain_sequences`.
    :type chain_sequences: Tuple[Tuple[str, str, str, str], str]
    :param blast_db: Location of BLAST database
    :type blast_db: str
    :param cutoff: Sequence identity cutoff; must be between 0 and 1
    :type cutoff: float
    :param num_alignments: Number of proteins for which to calculate identity.
    :type num_alignments: int
    
    :return: List of chain identifiers (as tuples) for proteins found within specified sequence identity.
    :rtype: List[Tuple[str, str, str, str]]
    """    
    """Find all other pdbs that have sequence identity greater than cutoff."""
    
    if 'BLAST_BIN' not in os.environ:
        raise RuntimeError('Need to set BLAST_BIN in .env to use blastp')
    if not (0 <= cutoff <= 1):
        raise Exception('cutoff need to be between 0 and 1')

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
            seq_id = float(nident)*1.0 / len(s)
            if seq_id >= cutoff:
                sim.add(_fasta_name_to_tuple(match))
    return list(sim)


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
    logger.info('getting clusters from PDB...')
    if not os.path.exists(f'bc-{id_level}.out'):
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


def write_to_blast_db(chain_sequences, blast_db):
    """Write provided sequences to blast db, for use with BLAST. Sequences should be provided as (id, sequence) tuples as returned by :func:`get_chain_sequences`.
    """
    logger.info(f'writing {len(chain_sequences):} entries to BLAST db {blast_db:}')

    if 'BLAST_BIN' not in os.environ:
        raise RuntimeError('Need to set BLAST_BIN in .env to use makeblastdb')

    if os.path.exists(blast_db):
        logger.warning(f'Removing previous blast db at {blast_db:}...')
        os.remove(blast_db)
    _write_fasta(chain_sequences, blast_db)
    cmd = os.path.join(os.environ['BLAST_BIN'], 'makeblastdb')
    subprocess.check_output(f'{cmd:} -in {blast_db:} -dbtype prot', shell=True)


def _tuple_to_fasta_name(name_tuple):
    """Fasta names can only be strings. This code writes the tuples in."""
    return '___'.join([str(x) for x in name_tuple])


def _fasta_name_to_tuple(x):
    """Fasta names can only be strings. This code gets the tuples back out."""
    return tuple(x.split('___'))


def get_chain_sequences(df):
    """Return list of tuples of (id, sequence) for different chains of monomers in a given dataframe."""
    # Keep only CA of standard residues
    df = df[df['name'] == 'CA'].drop_duplicates()
    df = df[df['resname'].apply(lambda x: Poly.is_aa(x, standard=True))]
    df['resname'] = df['resname'].apply(Poly.three_to_one)
    chain_sequences = []
    for c, chain in df.groupby(['ensemble', 'subunit', 'structure', 'model', 'chain']):
        seq = ''.join(chain['resname'])
        chain_sequences.append((tuple([str(x) for x in c]), seq))
    return chain_sequences


def _write_fasta(chain_sequences, outfile):
    """
    Write chain_sequences to fasta file. 
    """
    with open(outfile, 'w') as f:
        for chain, seq in chain_sequences:
            f.write('>' + _tuple_to_fasta_name(chain) + '\n')
            f.write(seq + '\n')