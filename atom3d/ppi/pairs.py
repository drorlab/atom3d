"""Code to generate pair ensembles."""
import click
import pandas as pd
import parallel as par

import atom3d.ppi.neighbors as nb
import atom3d.util.log as log
import atom3d.util.shard as sh
import atom3d.util.shard_ops as sho


logger = log.getLogger('shard_pairs')


@click.command(help='Generate interacting pairs from sharded dataset')
@click.argument('input_path', type=click.Path())
@click.argument('output_path', type=click.Path())
@click.option('-c', '--cutoff', type=int, default=8,
              help='Maximum distance (in angstroms), for two residues to be '
              'considered neighbors.')
@click.option('--cutoff-type', default='CA',
              type=click.Choice(['heavy', 'CA'], case_sensitive=False),
              help='How to compute distance between residues: CA is based on '
              'alpha-carbons, heavy is based on any heavy atom.')
@click.option('-n', '--num_threads', default=8,
              help='Number of threads to use for parallel processing.')
def shard_pairs(input_path, output_path, cutoff, cutoff_type,
                num_threads):
    input_sharded = sh.load_sharded(input_path)
    output_sharded = sh.Sharded(output_path, input_sharded.get_keys())
    input_num_shards = input_sharded.get_num_shards()

    tmp_path = output_sharded.get_prefix() + f'_tmp@{input_num_shards:}'
    tmp_sharded = sh.Sharded(tmp_path, input_sharded.get_keys())

    logger.info(f'Using {num_threads:} threads')

    inputs = [(input_sharded, tmp_sharded, shard_num, cutoff, cutoff_type)
              for shard_num in range(input_num_shards)]

    par.submit_jobs(_shard_pairs, inputs, num_threads)

    sho.reshard(tmp_sharded, output_sharded)
    tmp_sharded.delete_files()


def _shard_pairs(input_sharded, output_sharded, shard_num, cutoff,
                 cutoff_type):
    logger.info(f'Processing shard {shard_num:}')
    shard = input_sharded.read_shard(shard_num)
    num_structures = len(shard['ensemble'].unique())

    pairs = []
    for structure, x in shard.groupby('ensemble'):

        if len(x['subunit'].unique()) > 1:
            raise RuntimeError('Cannot find pairs on existing ensemble')
        # Only keep first model.
        x = x[x['model'] == sorted(x['model'].unique())[0]]
        names, subunits = _gen_subunits(x)

        for i in range(len(subunits)):
            for j in range(i + 1, len(subunits)):
                curr = nb.get_neighbors(
                    subunits[i], subunits[j], cutoff, cutoff_type)
                if len(curr) > 0:
                    tmp0 = subunits[i].copy()
                    tmp0['subunit'] = names[i]
                    tmp1 = subunits[j].copy()
                    tmp1['subunit'] = names[j]
                    pair = pd.concat([tmp0, tmp1])
                    pair['ensemble'] = names[i] + '_' + names[j]
                    pairs.append(pair)
    pairs = pd.concat(pairs).reset_index(drop=True)
    num_pairs = len(pairs['ensemble'].unique())
    output_sharded._write_shard(shard_num, pairs)
    logger.info(f'Done processing shard {shard_num:}, generated {num_pairs:} '
                f'pairs from {num_structures:} structures.')


def _gen_subunits(df):
    """Extract subunits to define protein interfaces for."""
    names = []

    subunits = []
    for name, x in df.groupby(['structure', 'model', 'chain']):
        names.append('_'.join([str(x) for x in name]))
        subunits.append(x)
    return names, subunits


if __name__ == "__main__":
    shard_pairs()
