"""Generate protein interfaces labels for DIPS dataset."""
import logging
import os
import timeit

import click
import parallel as par

import atom3d.ppi.gen_labels as gl
import atom3d.util.datatypes as dt
import atom3d.util.file as fi


@click.command(help='Find neighbors for DIPS dataset.')
@click.argument('path_to_dips', type=click.Path(exists=True))
@click.option('-c', '--cutoff', type=int, default=8,
              help='Maximum distance (in angstroms), for two residues to be '
              'considered neighbors.')
@click.option('--cutoff-type', default='CA',
              type=click.Choice(['heavy', 'CA'], case_sensitive=False),
              help='How to compute distance between residues: CA is based on '
              'alpha-carbons, heavy is based on any heavy atom.')
@click.option('-n', '--num_threads', default=8,
              help='Number of threads to use for parallel processing.')
def gen_labels_dips(path_to_dips, cutoff, cutoff_type, num_threads):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    requested_files = fi.find_files(path_to_dips, 'mmcif', relative=True)
    requested_keys = set([os.path.splitext(x)[0] for x in requested_files])

    produced_files = fi.find_files(
        path_to_dips, 'labels', relative=True)
    produced_keys = set([os.path.splitext(x)[0] for x in produced_files])

    work_keys = requested_keys.difference(produced_keys)
    work_files = [os.path.join(path_to_dips, key) for key in work_keys]

    logging.info(f'{len(requested_keys):} requested, '
                 f'{len(produced_keys):} already produced, '
                 f'{len(work_keys):} left to do.')

    inputs = [(f, cutoff, cutoff_type) for f in work_files]

    par.submit_jobs(_gen_labels_dips_single, inputs, num_threads)


def _gen_labels_dips_single(f, cutoff, cutoff_type):
    pdb_name = os.path.basename(f)
    logging.info(f'Processing {f:}')

    start_time = timeit.default_timer()
    start_time_reading = timeit.default_timer()
    df = dt.bp_to_df(dt.read_mmcif(f + '.mmcif'))
    elapsed_reading = timeit.default_timer() - start_time_reading

    start_time_processing = timeit.default_timer()
    # Keep only first model.
    df = df[df['model'] == 1]
    neighbors = gl.get_all_neighbors(
        [df], [], cutoff, cutoff_type)
    elapsed_processing = timeit.default_timer() - start_time_processing

    start_time_writing = timeit.default_timer()
    output_file = f + '.labels'
    neighbors.to_csv(output_file, index=False)
    elapsed_writing = timeit.default_timer() - start_time_writing
    elapsed = timeit.default_timer() - start_time

    logging.info(
        (f'For {len(neighbors):} neighbors extracted from {pdb_name:} '
            f'spent {elapsed_reading:05.2f} reading, '
            f'{elapsed_processing:05.2f} processing, '
            f'{elapsed_writing:05.2f} writing, '
            f'and {elapsed:05.2f} overall.'))


if __name__ == "__main__":
    gen_labels_dips()
