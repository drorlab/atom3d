"""Label structures with deviation."""
import click
import pandas as pd
import parallel as par

import atom3d.shard.shard as sh
import atom3d.util.log as log

logger = log.get_logger('rsr_label')


@click.command(help='Label QM9 structures.')
@click.argument('sharded_path', type=click.Path())
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('-n', '--num_threads', default=8,
              help='Number of threads to use for parallel processing.')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Overwrite existing labels.')

def gen_labels_sharded(sharded_path, csv_file, num_threads, overwrite):
    sharded = sh.Sharded.load(sharded_path)
    num_shards = sharded.get_num_shards()

    requested_shards = list(range(num_shards))
    if not overwrite:
        produced_shards = [x for x in requested_shards
                           if sharded.has(x, 'labels')]
    else:
        produced_shards = []

    labels_data = pd.read_csv(csv_file)

    work_shards = set(requested_shards).difference(produced_shards)
    logger.info(f'{len(requested_shards):} requested, '
                f'{len(produced_shards):} already produced, '
                f'{len(work_shards):} left to do.')
    logger.info(f'Using {num_threads:} threads')

    inputs = [(sharded, shard_num, labels_data)
              for shard_num in work_shards]

    par.submit_jobs(_gen_labels_shard, inputs, num_threads)


def _gen_labels_shard(sharded, shard_num, labels_data):
    
    logger.info(f'Processing shard {shard_num:}')
    
    shard = sharded.read_shard(shard_num)
    
    # Names of the labels (first column is mol_id and excluded)
    data_keys = labels_data.keys()[1:]
    
    # Create a list of all labels and make it a data frame
    all_labels = []
    for s, subunit in shard.groupby('subunit'):
        svalues = labels_data[labels_data['mol_id']==s][data_keys].values.tolist()[0]
        all_labels.append([s, s] + svalues)
    col_titles = ['ensemble', 'subunit'] + data_keys.tolist()
    all_labels = pd.DataFrame(all_labels, columns=col_titles)
    
    # Add labels to the shard
    sharded.add_to_shard(shard_num, all_labels, 'labels')
    
    logger.info(f'Done processing shard {shard_num:}')


if __name__ == "__main__":
    gen_labels_sharded()