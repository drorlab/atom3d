"""Label structures with deviation."""
import click
import pandas as pd
import parallel as par

import atom3d.datasets.rsr.score as sc
import atom3d.util.log as log
import atom3d.shard.shard as sh

logger = log.get_logger('rsr_label')


@click.command(help='Label RSP structures with RMSD to native.')
@click.argument('sharded_path', type=click.Path())
@click.argument('score_dir', type=click.Path(exists=True))
@click.option('-n', '--num_threads', default=8,
              help='Number of threads to use for parallel processing.')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Overwrite existing labels.')
def gen_labels_sharded(sharded_path, score_dir, num_threads, overwrite):
    sharded = sh.Sharded.load(sharded_path)
    num_shards = sharded.get_num_shards()

    requested_shards = list(range(num_shards))
    if not overwrite:
        produced_shards = [x for x in requested_shards
                           if sharded.has(x, 'labels')]
    else:
        produced_shards = []

    scores = sc.load_scores(score_dir)

    work_shards = set(requested_shards).difference(produced_shards)
    logger.info(f'{len(requested_shards):} requested, '
                f'{len(produced_shards):} already produced, '
                f'{len(work_shards):} left to do.')
    logger.info(f'Using {num_threads:} threads')

    inputs = [(sharded, shard_num, scores)
              for shard_num in work_shards]

    par.submit_jobs(_gen_labels_shard, inputs, num_threads)


def _gen_labels_shard(sharded, shard_num, scores):
    logger.info(f'Processing shard {shard_num:}')

    shard = sharded.read_shard(shard_num)

    all_labels = []
    for (e, s), subunit in shard.groupby(['ensemble', 'subunit']):
        all_labels.append((e, s, scores[e].loc[s]['rms']))
    all_labels = pd.DataFrame(
        all_labels, columns=['ensemble', 'subunit', 'label'])
    sharded.add_to_shard(shard_num, all_labels, 'labels')
    logger.info(f'Done processing shard {shard_num:}')


if __name__ == "__main__":
    gen_labels_sharded()
