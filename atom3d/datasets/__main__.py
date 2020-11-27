import click
import logging
import sys

import atom3d.datasets.datasets as da
import atom3d.datasets.scores as sc
import atom3d.util.file as fi
import atom3d.util.formats as fo

logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_lmdb', type=click.Path(exists=False))
@click.option('-f', '--filetype', type=click.Choice(['pdb', 'silent', 'xyz', 'xyz-gdb']),
              default='pdb')
@click.option('-sf', '--serialization_format',
              type=click.Choice(['msgpack', 'pkl', 'json']),
              default='json')
@click.option('--score_path', type=click.Path(exists=True))
def main(input_dir, output_lmdb, filetype, score_path, serialization_format):
    """Script wrapper to make_lmdb_dataset to create LMDB dataset."""
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    logger.info(f'filetype: {filetype}')
    if filetype == 'xyz-gdb':
        fileext = 'xyz'
    else:
        fileext = filetype
    file_list = fi.find_files(input_dir, fo.patterns[fileext])
    logger.info(f'Found {len(file_list)} files.')

    if score_path:
        logger.info('Looking up scores...')
        scores = sc.Scores(score_path)
        new_file_list = scores.remove_missing(file_list)
        logger.info(f'Keeping {len(new_file_list)} / {len(file_list)}')
        file_list = new_file_list
    else:
        scores = None

    da.make_lmdb_dataset(
        file_list, output_lmdb, filetype, scores, serialization_format)


if __name__ == "__main__":
    main()
