import click
import logging
import os
import subprocess

import numpy as np
import pandas as pd
import parallel as par

import atom3d.psp.util as util
import atom3d.util.file as fi


def run_tmscore_per_structure(tmscore_exe, decoy, native):
    '''
    Run TM-score to compare 'decoy' and 'native'. Return a tuple of
    rmsd, tm, gdt-ts, gdt-ha. Return None on error.
    '''
    try:
        output = subprocess.check_output(
            "{:} {:} {:}".format(tmscore_exe, decoy, native),
            universal_newlines=True,
            shell=True).split("\n")
        rmsd = float(output[14][-8:])
        tm, __, gdt_ts, gdt_ha = output[16:20]
        tm = float(tm[14:20])
        gdt_ts = float(gdt_ts[14:20])
        gdt_ha = float(gdt_ha[14:20])
        return rmsd, tm, gdt_ts, gdt_ha
    except Exception as e:
        logging.error("Failure when running TM-score on {:} with "
                      "error:\n{:}".format(decoy, e))
        return None


def run_tmscore_per_target(tmscore_exe, output_filename, target_name,
                           target_dir, struct_format):
    '''
    Run TM-score to compare all decoy structures of a target with its
    native structure. Write the result into a tab-delimited file with
    the following headers:
        <target>  <decoy>  <rmsd>  <tm_score>  <gdt_ts>  <gdt_ha>
    '''
    native = os.path.join(target_dir, '{:}.{:}'.format(
        target_name, struct_format))
    decoys = fi.find_files(target_dir, struct_format)
    logging.info("Running tm-scores for {:} with {:} decoys".format(
        target_name, len(decoys)))
    rows = []
    for decoy in decoys:
        result = run_tmscore_per_structure(tmscore_exe, decoy, native)
        if result == None:
            logging.warning("Skip target {:} decoy {:} due to failure".format(
                target_name, decoy))
            continue
        rmsd, tm, gdt_ts, gdt_ha = result
        rows.append([util.get_target_name(decoy), util.get_decoy_name(decoy),
                     rmsd, gdt_ts, gdt_ha, tm])
    df = pd.DataFrame(
        rows,
        columns=['target', 'decoy', 'rmsd', 'gdt_ts', 'gdt_ha', 'tm'])
    df = df.sort_values(
            ['rmsd', 'gdt_ts', 'gdt_ha', 'tm', 'decoy'],
            ascending=[True, False, False, False, False]).reset_index(drop=True)
    # Write to file
    df.to_csv(output_filename, sep='\t', index=False)


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('target_list', type=click.Path(exists=True))
@click.argument('labels_dir')
@click.option('--struct_format', '-ext', default='pdb')
@click.option('--num_cpus', '-c', default=1)
@click.option('--overwrite', '-ow', is_flag=True)
@click.option('--tmscore_exe', '-tm',
              default='/oak/stanford/groups/rondror/projects/atom3d/software/tmscore/TMscore')
def main(data_dir, target_list, labels_dir, struct_format,
         num_cpus, overwrite, tmscore_exe):
    """ Compute rmsd, tm-score, gdt-ts, gdt-ha of decoy structures
    """
    logger = logging.getLogger(__name__)
    logger.info("Compute rmsd, tm-score, gdt-ts, gdt-ha of decoys in {:}".format(
        data_dir))

    os.makedirs(labels_dir, exist_ok=True)

    with open(target_list, 'r') as f:
        requested_filenames = \
            [os.path.join(label_dir, '{:}.dat'.format(x.strip())) for x in f]
    logger.info("{:} requested keys".format(len(requested_filenames)))

    produced_filenames = []
    if not overwrite:
        produced_filenames = [f for f in fi.find_files(label_dir, 'dat') \
                              if 'targets' not in f]
    logger.info("{:} produced keys".format(len(produced_filenames)))

    inputs = []
    for filename in requested_filenames:
        if filename in produced_filenames:
            continue
        target_name = util.get_target_name(filename)
        target_dir = os.path.join(data_dir, target_name)
        inputs.append((tmscore_exe, filename, target_name,
                       target_dir, struct_format))

    logger.info("{:} work keys".format(len(inputs)))
    par.submit_jobs(run_tmscore_per_target, inputs, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
