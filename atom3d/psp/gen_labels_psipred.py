import click
import logging
import os
import subprocess
import time

import atom3d.util.file as fi
import numpy as np
import pandas as pd
import parallel as par


def __subprocess_run(cmd):
    try:
        subprocess.check_output(
            "{:} {:} {:}".format(tmscore_exe, decoy, native),
            shell=True)
    except Exception e:
        logger.error(e)


def run_psipred(path, target):
    tic = time.time()

    name = os.path.join(path, target, target)

    # Run BLAST to generate matrix of counts and positions
    logger.debug("{:}: Running BLAST...".format(target))
    cmd = "{}/blastpgp -j 3 -d {} -i {}.fasta -h 0.001 -C {}.chk -Q {}.pssm -o {}.blast".format(
        blast_path, nr_path, name, name, name, name)
    __subprocess_run(cmd)

    # Populate files as required
    cmd = "echo {}.fasta > {}.sn".format(target, name)
    __subprocess_run(cmd)
    cmd = "echo {}.chk > {}.pn".format(target, name)
    __subprocess_run(cmd)

    # Generate profile matrix
    logger.debug("{:}: Generating profile matrix...".format(target))
    cmd = "{}/makemat -P {}".format(blast_path, name)
    __subprocess_run(cmd)

    # PSIpred pass 1
    logger.debug("{:}: Running PSIpred first pass...".format(target))
    cmd = "{}/bin/psipred {}.mtx {}/data/weights.dat {}/data/weights.dat2 {}/data/weights.dat3 > {}.ss".\
        format(psipred_path, name, psipred_path, psipred_path, psipred_path, name)
    __subprocess_run(cmd)

    # PSIpred pass 2
    logger.debug("{:}: Running PSIpred second pass...".format(target))
    cmd = "{}/bin/psipass2 {}/data/weights_p2.dat 1 1.0 1.0 {}.ss2 {}.ss > {}.horiz".\
        format(psipred_path, psipred_path, name, name, name)
    __subprocess_run(cmd)

    # Remove unecessary files
    cmd = "rm {}.pn {}.sn {}.ss".format(name, name, name)
    __subprocess_run(cmd)

    elapsed = time.time() - tic
    logger.debug("{:}: Finished in {:.2f} sec ({:.2f} min".format(
        target, elapsed, elapsed/60.0))


@click.cmd()
@click.argument('path', type=click.Path(exists=True))
@click.argument('target_list', type=click.Path(exists=True))
@click.option('--num_cpus', '-c', default=1)
@click.option('--blast_path', '-bl',
              default='/oak/stanford/groups/rondror/users/bjing/bin/blast/bin')
@click.option('--nr_path', '-nr',
              default='/oak/stanford/groups/rondror/projects/ppi/non-redundant/nr')
@click.option('--psipred_path', '-psi',
              default='/oak/stanford/groups/rondror/users/bjing/bin/psipred')
### USAGE:
# frag.py <path-to-directory-containing-folders-for-each-target> <path-to-list-of-targets>
def main(path, target_list, num_cpus, blast_path, nr_path, psipred_path):
    """ Run psipreds to generate the PSSMs and secondary structure predictions
    for each residue in a protein structure.
    """
    logger = logging.getLogger(__name__)

    with open(target_list, 'r') as f:
        targets = [t.strip() for t in f.readlines()]
    logger.info("Running psipreds on {:} structures in {:}".format(
        len(targets), target_list))

    def __run(id, path, target):
        logger.info(">>>> Processing target {:} ({:}/{:})...".format(
            target, id, len(targets)))
        run_psipred(path, target)

    inputs = []
    for i, target in enumerate(targets):
        inputs.append((i+1, path, target))

    print("Setting up pool wih {:} cpus".format(num_cpus))
    par.submit_jobs(__run, inputs, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
