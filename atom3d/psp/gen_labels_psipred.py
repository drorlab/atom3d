import click
import logging
import os
import subprocess
import time

import Bio.SeqUtils

import numpy as np
import pandas as pd
import parallel as par

import atom3d.psp.util as util
import atom3d.util.file as fi


def __subprocess_run(cmd):
    try:
        logging.debug(cmd)
        subprocess.check_output(
            cmd,
            universal_newlines=True,
            shell=True)
    except Exception as e:
        logging.error(e)
        raise e


def parse_ss2_file(ss2_filename, secstructs_dir, target):
    df = pd.read_csv(ss2_filename, delimiter='\s*', skiprows=[0], header=None,
                     names=['residue', 'resname', 'ss', 'coil', 'helix', 'strand'],
                     engine='python').dropna()
    # Convert 1-letter to 3-letter AA code
    #df['resname'] = df['resname'].apply(lambda x: Bio.SeqUtils.seq3(x).upper())
    # Insert target name
    df.insert(0, 'target', target)
    # Write to file
    output_filename = os.path.join(secstructs_dir, '{}.dat'.format(target))
    logging.debug("{:}: Writing ss2 to {:}".format(target, output_filename))
    df.to_csv(output_filename, sep='\t', index=False, float_format='%.3f')


def parse_pssm_file(pssm_filename, pssms_dir, psfms_dir, target):
    '''
    Parse the .pssm output from PSI-BLAST into separate files containing the
    PSSM and PSFM.
    '''
    pssm = pd.read_table(
        pssm_filename, skiprows=2, skipfooter=6, delim_whitespace=True,
        engine='python', usecols=range(20), index_col=[0, 1])
    pssm = pssm.reset_index()
    pssm.rename(columns={'level_0': 'res_id', 'level_1': 'res'}, inplace=True)

    pscm = pd.read_table(
        pssm_filename, skiprows=2, skipfooter=6, delim_whitespace=True,
        engine='python', usecols=range(20, 40), index_col=[0, 1])
    psfm = pscm.applymap(lambda x: x / 100.)
    psfm = psfm.reset_index()
    psfm.rename(columns={'level_0': 'res_id', 'level_1': 'res'}, inplace=True)
    psfm.columns = pssm.columns

    pssm.insert(0, 'target', target)
    psfm.insert(0, 'target', target)
    # Write PSSM to file
    out_pssm = os.path.join(pssms_dir, '{}.dat'.format(target))
    logging.debug("{:}: Writing PSSM to {:}".format(target, out_pssm))
    pssm.to_csv(out_pssm, sep='\t', index=False)

    # Write PSFM to file
    out_psfm = os.path.join(psfms_dir, '{}.dat'.format(target))
    logging.debug("{:}: Writing PSFM to {:}".format(target, out_psfm))
    psfm.to_csv(out_psfm, sep='\t', index=False)


def run_psipred(blast_path, nr_path, psipred_path, pssms_dir, psfms_dir,
                secstructs_dir, tmp_dir, target, fasta_file):
    tic = time.time()

    try:
        output_dir = os.path.join(tmp_dir, target)
        os.makedirs(output_dir, exist_ok=True)
        name = os.path.join(output_dir, target)

        # Run BLAST to generate matrix of counts and positions
        logging.debug("{:}: Running BLAST".format(target))
        cmd = "{}/blastpgp -j 3 -d {} -i {} -h 0.001 -C {}.chk -Q {}.pssm -o {}.blast".format(
            blast_path, nr_path, fasta_file, name, name, name)
        if not os.path.exists(name + ".chk"):
            __subprocess_run(cmd)

        # Populate files as required
        cmd = "cp {} {}.fasta; echo {}.fasta > {}.sn".format(
            fasta_file, name, target, name)
        __subprocess_run(cmd)
        cmd = "echo {}.chk > {}.pn".format(target, name)
        __subprocess_run(cmd)

        # Generate profile matrix
        logging.debug("{:}: Generating profile matrix".format(target))
        cmd = "{}/makemat -P {}".format(blast_path, name)
        __subprocess_run(cmd)

        # PSIpred pass 1
        logging.debug("{:}: Running PSIpred first pass".format(target))
        cmd = "{}/bin/psipred {}.mtx {}/data/weights.dat {}/data/weights.dat2 {}/data/weights.dat3 > {}.ss".format(
            psipred_path, name, psipred_path, psipred_path, psipred_path, name)
        __subprocess_run(cmd)

        # PSIpred pass 2
        logging.debug("{:}: Running PSIpred second pass".format(target))
        cmd = "{}/bin/psipass2 {}/data/weights_p2.dat 1 1.0 1.0 {}.ss2 {}.ss > {}.horiz".format(
            psipred_path, psipred_path, name, name, name)
        __subprocess_run(cmd)

        # Remove unecessary files
        cmd = "rm {}.pn {}.sn {}.ss".format(name, name, name)
        __subprocess_run(cmd)

        # Process the PSSM and ss2 outputs
        parse_pssm_file('{}.pssm'.format(name), pssms_dir, psfms_dir, target)
        parse_ss2_file('{}.ss2'.format(name), secstructs_dir, target)

    except Exception as e:
        logging.error(e)

    elapsed = time.time() - tic
    logging.info("{:}: Finished in {:.2f} sec ({:.2f} min)".format(
        target, elapsed, elapsed/60.0))


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--fastas_dir', '-fastas', default='fastas') # Relative to data_dir
@click.option('--pssms_dir', '-pssm', default='labels/pssms') # Relative to data_dir
@click.option('--psfms_dir', '-psfm', default='labels/psfms') # Relative to data_dir
@click.option('--secstructs_dir', '-ss', default='labels/secstructs') # Relative to data_dir
@click.option('--target_list', '-target', default='targets.dat') # Relative to data_dir
@click.option('--num_cpus', '-c', default=1)
@click.option('--blast_path', '-bl',
              default='/oak/stanford/groups/rondror/users/bjing/bin/blast/bin')
@click.option('--nr_path', '-nr',
              default='/oak/stanford/groups/rondror/projects/ppi/non-redundant/nr')
@click.option('--psipred_path', '-psi',
              default='/oak/stanford/groups/rondror/users/bjing/bin/psipred')
def main(data_dir, fastas_dir, pssms_dir, psfms_dir, secstructs_dir,
         target_list, num_cpus, blast_path, nr_path, psipred_path):
    """ Run psipreds to generate the PSSMs and secondary structure predictions
    for each residue in a protein structure.
    """
    logger = logging.getLogger(__name__)

    with open(os.path.join(data_dir, target_list), 'r') as f:
        targets = [t.strip() for t in f.readlines()]

    logger.info("Running psipreds on {:} structures in {:}".format(
        len(targets), target_list))

    def __run(id, blast_path, nr_path, psipred_path, pssms_dir, psfms_dir,
              secstructs_dir, tmp_dir, target, fasta_file):
        logger.info("Processing target {:} ({:}/{:})".format(
            target, id, len(targets)))
        run_psipred(blast_path, nr_path, psipred_path, pssms_dir, psfms_dir,
                    secstructs_dir, tmp_dir, target, fasta_file)

    pssms_dir = os.path.join(data_dir, pssms_dir)
    psfms_dir = os.path.join(data_dir, psfms_dir)
    secstructs_dir = os.path.join(data_dir, secstructs_dir)
    tmp_dir = os.path.join(data_dir, '.tmp')

    os.makedirs(pssms_dir, exist_ok=True)
    os.makedirs(psfms_dir, exist_ok=True)
    os.makedirs(secstructs_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    inputs = []
    for i, target in enumerate(targets):
        fasta_file = os.path.join(
            data_dir, fastas_dir, '{:}.fasta'.format(target))
        if os.path.exists(fasta_file):
            inputs.append((i+1, blast_path, nr_path, psipred_path, pssms_dir,
                           psfms_dir, secstructs_dir, tmp_dir, target, fasta_file))
        else:
            logger.warning("FASTA for {:} does not exist".format(target))

    par.submit_jobs(__run, inputs, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    #logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    main()
