import torch
import atom3d.datasets.datasets as da
import click
import logging
import os

logger = logging.getLogger(__name__)

def read_split_file(txt):
    split_indices = []
    with open(txt) as f:
        for line in f:
            split_indices.append(int(line.strip()))
    return split_indices
    
@click.command(help='Split dataset')
@click.argument('in_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--train_txt', '-tr', type=click.Path(exists=True), default=None)
@click.option('--val_txt', '-v', type=click.Path(exists=True), default=None)
@click.option('--test_txt', '-t', type=click.Path(exists=True), default=None)
def split(in_path, output_root, train_txt, val_txt, test_txt):
    dataset = da.load_dataset(in_path, 'lmdb')

    logger.info(f'Writing train')
    train_indices = read_split_file(train_txt)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    da.make_lmdb_dataset(train_dataset, os.path.join(output_root, 'train'))

    logger.info(f'Writing val')
    val_indices = read_split_file(val_txt)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    da.make_lmdb_dataset(val_dataset, os.path.join(output_root, 'val'))

    logger.info(f'Writing test')
    test_indices = read_split_file(test_txt)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    da.make_lmdb_dataset(test_dataset, os.path.join(output_root, 'test'))


if __name__ == "__main__":
    split()