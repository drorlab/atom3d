"""
This script will combine multiple LMDBs in a directory into a single LMDB dataset.
"""
import click
import sys
import pandas as pd
import lmdb
import io
import gzip
import logging
import tqdm
from atom3d.datasets.datasets import LMDBDataset, make_lmdb_dataset, serialize

logger = logging.getLogger(__name__)

@click.command()
@click.argument('lmdb_list', nargs=-1)
@click.argument('output_lmdb', type=click.Path(exists=False))
@click.option('--append', '-a',  is_flag=True)
def main(lmdb_list, output_lmdb, append):
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                       level=logging.INFO)
    env = lmdb.open(str(output_lmdb), map_size=int(1e12))
    max_i = 0
    if append:
        for key, value in env.cursor():
            max_i = max(max_i, key)
    
    with env.begin(write=True) as txn:
        id_to_idx = {}
        i = max_i + 1
        for db_idx, db in enumerate(lmdb_list):
            logger.info(f'on database {db_idx + 1} of {len(lmdb_list)}')
            
            dataset = LMDBDataset(db)
            num_examples = dataset._num_examples
            serialization_format = dataset._serialization_format

        
            for x in tqdm.tqdm(dataset, total=num_examples):
                # Add an entry that stores the original types of all entries
                x['types'] = {key: str(type(val)) for key, val in x.items()}
                # ... including itself
                x['types']['types'] = str(type(x['types']))
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                    f.write(serialize(x, serialization_format))
                compressed = buf.getvalue()
                result = txn.put(str(i).encode(), compressed, overwrite=False)
                if not result:
                    raise RuntimeError(f'LMDB entry {i} in {str(output_lmdb)} '
                                    'already exists')
                id_to_idx[x['id']] = i
                i += 1

        txn.put(b'num_examples', str(i).encode())
        txn.put(b'serialization_format', serialization_format.encode())
        txn.put(b'id_to_idx', serialize(id_to_idx, serialization_format))

if __name__ == "__main__":
    main()
