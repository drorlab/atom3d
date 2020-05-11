"""Code for preparing a lap dataset (filtering and splitting)."""
import click

import atom3d.util.filters as filters
import atom3d.util.shard as sh
import atom3d.util.shard_ops as sho
import atom3d.util.splits as splits


def split(input_sharded, output_root):
    """Split randomly."""

    ensembles = sh.get_names(input_sharded)
    num_total = len(ensembles)

    indices_test, indices_val, indices_train = splits.random_split(num_total)

    # Will just look up ensembles.
    train = [ensembles[i] for i in indices_train]
    val = [ensembles[i] for i in indices_val]
    test = [ensembles[i] for i in indices_test]

    prefix = sh._get_prefix(output_root)
    num_shards = sh.get_num_shards(output_root)
    train_sharded = f'{prefix:}_train@{num_shards:}'
    val_sharded = f'{prefix:}_val@{num_shards:}'
    test_sharded = f'{prefix:}_test@{num_shards:}'

    train_filter_fn = filters.form_filter_against_list(train, 'ensemble')
    val_filter_fn = filters.form_filter_against_list(val, 'ensemble')
    test_filter_fn = filters.form_filter_against_list(test, 'ensemble')

    sho.filter_sharded(input_sharded, train_sharded, train_filter_fn)
    sho.filter_sharded(input_sharded, val_sharded, val_filter_fn)
    sho.filter_sharded(input_sharded, test_sharded, test_filter_fn)


@click.command(help='Prepare lap dataset')
@click.argument('input_sharded', type=click.Path())
@click.argument('output_sharded', type=click.Path())
def prepare(input_sharded, output_sharded):
    split(input_sharded, output_sharded)


if __name__ == "__main__":
    prepare()
