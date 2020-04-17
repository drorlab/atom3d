"""Utility to convert whole dataset from pdb to mmcif."""
import logging
import os

import click
import tqdm

import atom3d.util.datatypes as dt
import atom3d.util.file as fi




@click.command(help='Convert whole dataset from pdbs to mmcif')
@click.argument('path_to_input', type=click.Path(exists=True))
@click.argument('path_to_output', type=click.Path())
@click.option('--remove-ext/--keep-ext', default=True,
              help='whether to keep old file extension or append to.')
def convert_pdb_to_mmcif(path_to_input, path_to_output, remove_ext):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    requested_files = \
        fi.find_files(path_to_input, 'pdb[0-9]*\.gz', relative=True) + \
        fi.find_files(path_to_input, 'pdb[0-9]*', relative=True)

    produced_files = fi.find_files(path_to_output, 'mmcif', relative=True)

    def output_from_input(x):
        """Get output file name from input file name."""
        if remove_ext:
            if len(x) >= 3 and x[-3:] == '.gz':
                # Remove .gz
                x = os.path.splitext(x)[0]
            x = os.path.splitext(x)[0]
        x += '.mmcif'
        return x

    work_inputs = [(x, output_from_input(x)) for x in requested_files
                   if output_from_input(x) not in set(produced_files)]

    logging.info(f'{len(requested_files):} requested, '
                 f'{len(produced_files):} already converted, '
                 f'{len(work_inputs):} left to do.')

    for input_file, output_file in tqdm.tqdm(work_inputs):
        input_path = os.path.join(path_to_input, input_file)
        output_path = os.path.join(path_to_output, output_file)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        dt.write_mmcif(output_path, dt.read_pdb(input_path))


if __name__ == "__main__":
    convert_pdb_to_mmcif()
