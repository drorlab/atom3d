from atom3d.datasets import load_dataset, make_lmdb_dataset
import sys

in_path = sys.argv[1]
out_path = sys.argv[2]

dataset = load_dataset(in_path, 'lmdb')

make_lmdb_dataset(dataset, out_path, filter_fn=lambda x: not x)
