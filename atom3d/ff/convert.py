import os

import click
import tqdm

import atom3d.util.datatypes as dt
import atom3d.util.shard as sh


@click.command(help='Sample pdbs from sharded')
@click.argument('input_sharded', type=click.Path())
@click.argument('output_dir', type=click.Path())
@click.argument('num', type=int)
def convert(input_sharded, output_dir, num):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_sharded = sh.load_sharded(input_sharded)

    names = input_sharded.get_names().sample(num)
    keys = input_sharded.get_keys()
    for _, key in tqdm.tqdm(names[keys].iterrows(), total=names.shape[0]):
        df = input_sharded.read_keyed(key)
        bp = dt.df_to_bp(df)
        # Remove any extensions.
        name = '.'.join(key[0].split('.')[:2])
        out_file = os.path.join(output_dir, name)
        dt.write_pdb(out_file, bp)


if __name__ == "__main__":
    convert()
