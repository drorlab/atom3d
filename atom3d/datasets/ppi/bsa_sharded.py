"""Generate BSA database for sharded dataset."""
import multiprocessing as mp
import os
import timeit

import click
import pandas as pd
import parallel as par

import atom3d.datasets.ppi.bsa as bsa
import atom3d.datasets.ppi.neighbors as nb
import atom3d.util.log as log
import atom3d.shard.shard as sh

logger = log.get_logger('bsa')

db_sem = mp.Semaphore()


@click.command(help='Generate Buried Surface Area database for sharded.')
@click.argument('sharded_path', type=click.Path())
@click.argument('output_bsa', type=click.Path())
@click.option('-n', '--num_threads', default=8,
              help='Number of threads to use for parallel processing.')
def bsa_db(sharded_path, output_bsa, num_threads):
    sharded = sh.Sharded.load(sharded_path)
    num_shards = sharded.get_num_shards()

    dirname = os.path.dirname(output_bsa)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)

    inputs = [(sharded, x, output_bsa) for x in range(num_shards)]
    logger.info(f'{num_shards:} shards to do.')
    logger.info(f'Using {num_threads:} threads')

    par.submit_jobs(_bsa_db, inputs, num_threads)


def _bsa_db(sharded, shard_num, output_bsa):
    logger.info(f'Processing shard {shard_num:}')
    start_time = timeit.default_timer()
    start_time_reading = timeit.default_timer()
    shard = sharded.read_shard(shard_num)
    elapsed_reading = timeit.default_timer() - start_time_reading

    start_time_waiting = timeit.default_timer()
    with db_sem:
        start_time_reading = timeit.default_timer()
        if os.path.exists(output_bsa):
            curr_bsa_db = pd.read_csv(output_bsa).set_index(['ensemble'])
        else:
            curr_bsa_db = None
        tmp_elapsed_reading = timeit.default_timer() - start_time_reading
    elapsed_waiting = timeit.default_timer() - start_time_waiting - \
        tmp_elapsed_reading
    elapsed_reading += tmp_elapsed_reading

    start_time_processing = timeit.default_timer()
    all_results = []
    cache = {}
    for e, ensemble in shard.groupby('ensemble'):
        if (curr_bsa_db is not None) and (e in curr_bsa_db.index):
            continue
        (name0, name1, _, _), (bdf0, bdf1, _, _) = nb.get_subunits(ensemble)

        try:
            # We use bound for indiviudal subunits in bsa computation, as
            # sometimes the actual structure between bound and unbound differ.
            if name0 not in cache:
                cache[name0] = bsa._compute_asa(bdf0)
            if name1 not in cache:
                cache[name1] = bsa._compute_asa(bdf1)
            result = bsa.compute_bsa(bdf0, bdf1, cache[name0], cache[name1])
            result['ensemble'] = e
            all_results.append(result)
        except AssertionError as e:
            logger.warning(e)
            logger.warning(f'Failed BSA on {e:}')

    if len(all_results) > 0:
        to_add = pd.concat(all_results, axis=1).T
    elapsed_processing = timeit.default_timer() - start_time_processing

    if len(all_results) > 0:
        start_time_waiting = timeit.default_timer()
        with db_sem:
            start_time_writing = timeit.default_timer()
            # Update db in case it has updated since last run.
            if os.path.exists(output_bsa):
                curr_bsa_db = pd.read_csv(output_bsa)
                new_bsa_db = pd.concat([curr_bsa_db, to_add])
            else:
                new_bsa_db = to_add
            new_bsa_db.to_csv(output_bsa + f'.tmp{shard_num:}', index=False)
            os.rename(output_bsa + f'.tmp{shard_num:}', output_bsa)
            elapsed_writing = timeit.default_timer() - start_time_writing
        elapsed_waiting += timeit.default_timer() - start_time_waiting - \
            elapsed_writing
    else:
        elapsed_writing = 0
    elapsed = timeit.default_timer() - start_time

    logger.info(
        f'For {len(all_results):03d} pairs buried in shard {shard_num:} spent '
        f'{elapsed_reading:05.2f} reading, '
        f'{elapsed_processing:05.2f} processing, '
        f'{elapsed_writing:05.2f} writing, '
        f'{elapsed_waiting:05.2f} waiting, and '
        f'{elapsed:05.2f} overall.')


if __name__ == "__main__":
    bsa_db()
