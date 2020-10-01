import glob
import click
import logging

from . import datasets as da
from . import scores as sc
import atom3d.util.file as fi
import atom3d.util.formats as ft

logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_lmdb', type=click.Path(exists=False))
@click.option('--filetype', type=click.Choice(['pdb', 'silent', 'xyz', 'xyz-gdb']),
              default='pdb')
@click.option('-sf', '--serialization_format',
              type=click.Choice(['msgpack', 'pkl', 'json']),
              default='json')
@click.option('--score_path', type=click.Path(exists=True))
def main(input_dir, output_lmdb, filetype, score_path, serialization_format):
    """Script wrapper to make_lmdb_dataset to create LMDB dataset."""
    print('filetype:', filetype)
    if filetype == 'pdb':
        file_list = fi.find_files(input_dir, ft.patterns['pdb']) + \
            fi.find_files(input_dir, ft.patterns['pdb.gz'])
    elif filetype in ['xyz', 'xyz-gdb']:
        file_list = glob.glob(input_dir+'*.xyz')
        print('Found %i XYZ files.'%(len(file_list)))
    else:
        file_list = fi.find_files(input_dir, '.out')

    if score_path:
        scores = sc.Scores(score_path)
        new_file_list = scores.remove_missing(file_list)
        logger.info(f'Keeping {len(new_file_list)} / {len(file_list)}')
        input_data_path = new_file_list
    else:
        scores = None

    da.make_lmdb_dataset(
        file_list, output_lmdb, filetype, scores, serialization_format)


if __name__ == "__main__":
    main()

