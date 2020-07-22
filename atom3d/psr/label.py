"""Label structures with RMSD, GDT_TS, GDT_HA, TM-score."""
import click
import numpy as np
import pandas as pd
import parallel as par

import atom3d.psr.util as psr_util
import atom3d.util.log as log
import atom3d.shard.shard as sh

logger = log.get_logger('psr_label')


@click.command(help='Label RSP structures with RMSD, GDT_TS, GDT_HA, TM-label to native.')
@click.argument('sharded_path', type=click.Path())
@click.argument('label_dir', type=click.Path(exists=True))
@click.option('-n', '--num_threads', default=8,
              help='Number of threads to use for parallel processing.')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Overwrite existing labels.')
def gen_labels_sharded(sharded_path, label_dir, num_threads, overwrite):
    sharded = sh.Sharded.load(sharded_path)
    num_shards = sharded.get_num_shards()

    requested_shards = list(range(num_shards))
    if not overwrite:
        produced_shards = [x for x in requested_shards
                           if sharded.has(x, 'labels')]
    else:
        produced_shards = []

    labels = psr_util.read_labels(label_dir)

    work_shards = set(requested_shards).difference(produced_shards)
    logger.info(f'{len(requested_shards):} requested, '
                f'{len(produced_shards):} already produced, '
                f'{len(work_shards):} left to do.')
    logger.info(f'Using {num_threads:} threads')

    inputs = [(sharded, shard_num, labels)
              for shard_num in work_shards]

    par.submit_jobs(_gen_labels_shard, inputs, num_threads)


def _gen_labels_shard(sharded, shard_num, labels):
    logger.info(f'Processing shard {shard_num:}')

    shard = sharded.read_shard(shard_num)

    all_labels = []
    for (e, s), subunit in shard.groupby(['ensemble', 'subunit']):
        row = {'ensemble': e, 'subunit': s}
        try:
            row.update(labels.loc[(e, s)].to_dict())
        except:
            row.update(dict.fromkeys(labels.columns.values, np.nan))
            logger.error(f'Missing label for {e:}/{s:} to process shard {shard_num:}')
        all_labels.append(row)
    all_labels = pd.DataFrame(all_labels)
    sharded.add_to_shard(shard_num, all_labels, 'labels')
    logger.info(f'Done processing shard {shard_num:}')


if __name__ == "__main__":
    gen_labels_sharded()
