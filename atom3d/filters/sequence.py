import pandas as pd

import atom3d.protein.sequence as seq


def form_seq_filter_against(sharded, cutoff):
    """
    Remove structures with too much sequence identity to a chain in sharded.

    We consider each chain in each structure separately, and remove the
    structure if any of them matches any chain in sharded.
    """
    blast_db_path = f'{sharded.path:}.db'
    all_chain_sequences = []
    for _, shard in sharded.iter_shards():
        all_chain_sequences.extend(seq.get_all_chain_sequences_df(shard))
    seq.write_to_blast_db(all_chain_sequences, blast_db_path)

    def filter_fn(df):
        to_keep = {}
        for structure_name, cs in seq.get_all_chain_sequences_df(df):
            hits = seq.find_similar(cs, blast_db_path, cutoff, 1)
            ensemble = structure_name[0]
            to_keep[ensemble] = (len(hits) == 0)
        to_keep = pd.Series(to_keep)[df['ensemble']]
        return df[to_keep.values]

    return filter_fn