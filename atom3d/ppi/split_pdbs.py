"""Methods to compare DIPS pair dills to mmcif files."""

import click

import atom3d.util.datatypes as dt

@click.command()
@click.argument('input_pdb', type=click.Path(exists=True))
@click.argument('output_prefix', type=click.Path())
def split_pdb(input_pdb, output_prefix):
    bp = dt.read_pdb(input_pdb)
    df = dt.bp_to_df(bp)
    print(df)


if __name__ == "__main__":
    split_pdb()
