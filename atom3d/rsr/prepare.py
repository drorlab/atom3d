"""Code for preparing rsr dataset (splitting)."""
import click

import atom3d.rsr.score as sc
import atom3d.util.filters as filters
import atom3d.util.log as log
import atom3d.util.shard as sh
import atom3d.util.shard_ops as sho


logger = log.getLogger('rsr_prepare')


# Canonical splits.
TRAIN = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
VAL = ['14b', '14f', '15', '17']
TEST = ['18', '19', '20', '21']


def split(input_sharded, output_root):
    """Split temporally and shuffle examples."""
    prefix = sh.get_prefix(output_root)
    num_shards = sh.get_num_shards(output_root)

    # Re-key to ensemble, subunit.  To allow for shuffling across targets.
    tmp_sharded = sh.Sharded(
        f'{prefix:}_rekeyed@{num_shards:}', ['ensemble', 'subunit'])
    logger.info("Rekeying")
    sho.rekey(input_sharded, tmp_sharded)

    keys = tmp_sharded.get_keys()
    train_sharded = sh.Sharded(f'{prefix:}_train@{num_shards:}', keys)
    val_sharded = sh.Sharded(f'{prefix:}_val@{num_shards:}', keys)
    test_sharded = sh.Sharded(f'{prefix:}_test@{num_shards:}', keys)

    logger.info("Splitting")
    train_filter_fn = filters.form_filter_against_list(TRAIN, 'ensemble')
    val_filter_fn = filters.form_filter_against_list(VAL, 'ensemble')
    test_filter_fn = filters.form_filter_against_list(TEST, 'ensemble')

    sho.filter_sharded(
        tmp_sharded, train_sharded, train_filter_fn, num_shards)
    sho.filter_sharded(
        tmp_sharded, val_sharded, val_filter_fn, num_shards)
    sho.filter_sharded(
        tmp_sharded, test_sharded, test_filter_fn, num_shards)

    tmp_sharded.delete_files()


@click.command(help='Prepare rsr dataset')
@click.argument('input_sharded_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--score_dir', type=click.Path(exists=True), default=None)
def prepare(input_sharded_path, output_root, score_dir):
    input_sharded = sh.load_sharded(input_sharded_path)
    if score_dir is not None:
        prefix = sh.get_prefix(output_root)
        num_shards = sh.get_num_shards(output_root)
        keys = input_sharded.get_keys()
        filter_sharded = sh.Sharded(f'{prefix:}_filtered@{num_shards:}', keys)
        filter_fn = sc.form_score_filter(score_dir)
        logger.info('Filtering against score file.')
        sho.filter_sharded(input_sharded, filter_sharded, filter_fn)
        split(filter_sharded, output_root)
        filter_sharded.delete_files()
    else:
        split(input_sharded, output_root)


if __name__ == "__main__":
    prepare()
