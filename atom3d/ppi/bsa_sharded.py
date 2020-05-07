"""Generate BSA database for sharded dataset."""
import ast
import os
import timeit

import click
import multiprocessing as mp
import pandas as pd
import parallel as par

import atom3d.ppi.bsa as bsa
import atom3d.ppi.neighbors as nb
import atom3d.util.log as log
import atom3d.util.shard as sh


logger = log.getLogger('bsa')

db_sem = mp.Semaphore()


@click.command(help='Generate Buried Surface Area database for sharded.')
@click.argument('sharded', type=click.Path())
@click.argument('output_bsa', type=click.Path())
@click.option('-n', '--num_threads', default=8,
              help='Number of threads to use for parallel processing.')
def bsa_db(sharded, output_bsa, num_threads):
    num_shards = sh.get_num_shards(sharded)

    dirname = os.path.dirname(output_bsa)
    os.makedirs(dirname, exist_ok=True)

    inputs = [(sharded, x, output_bsa) for x in range(num_shards)]
    logger.info(f'{num_shards:} shards to do.')
    logger.info(f'Using {num_threads:} threads')

    par.submit_jobs(_bsa_db, inputs, num_threads)


def _bsa_db(sharded, shard_num, output_bsa):
    logger.info(f'Processing shard {shard_num:}')
    start_time = timeit.default_timer()
    start_time_reading = timeit.default_timer()
    shard = sh.read_shard(sharded, shard_num)
    pairs = sh.read_shard(sharded, shard_num, 'pairs')
    elapsed_reading = timeit.default_timer() - start_time_reading

    start_time_waiting = timeit.default_timer()
    with db_sem:
        start_time_reading = timeit.default_timer()
        if os.path.exists(output_bsa):
            curr_bsa_db = pd.read_csv(
                output_bsa, converters={
                    "subunit0": ast.literal_eval,
                    "subunit1": ast.literal_eval
                }).set_index(['subunit0', 'subunit1'])
        else:
            curr_bsa_db = None
        tmp_elapsed_reading = timeit.default_timer() - start_time_reading
    elapsed_waiting = timeit.default_timer() - start_time_waiting - \
        tmp_elapsed_reading
    elapsed_reading += tmp_elapsed_reading

    start_time_processing = timeit.default_timer()
    all_results = []
    cache = {}
    for i, (name0, name1) in enumerate(pairs):
        pair_name = (name0, name1)
        if (curr_bsa_db is not None) and (pair_name in curr_bsa_db.index):
            continue
        logger.info(f'{pair_name:}')
        subunit0 = nb.lookup_subunit(name0, shard)
        subunit1 = nb.lookup_subunit(name1, shard)

        # We use bound for indiviudal subunits in bsa computation, as sometimes
        # the actual structure between bound and unbound differ.
        if name0 not in cache:
            cache[name0] = bsa._compute_asa(subunit0['bound'])
        if name1 not in cache:
            cache[name1] = bsa._compute_asa(subunit1['bound'])
        all_results.append(bsa.compute_bsa(
            subunit0, subunit1, cache[name0], cache[name1]))

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
            new_bsa_db.to_csv(output_bsa, index=False)
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
