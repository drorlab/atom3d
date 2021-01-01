"""
Filtering functions for protein sequences.

"""
import pandas as pd

import atom3d.protein.sequence as seq


def form_seq_filter_against(dataset, cutoff):
    """
    Create filter that removes remove structures with too much sequence identity to a chain in supplied dataset.

    We consider each chain in each structure separately, and remove the structure if any of them matches any chain in dataset.

    :param dataset: dataset that if we are checking for matches against.
    :type dataset: atom3d dataset.
    :param cutoff: maximum allowable sequence identity to an entry in dataset before we filter.
    :type cutoff: float (0-1).

    :return: function that implements the specified filter.
    :rtype: filter function.
    """
    blast_db_path = f'blast_db'
    all_chain_sequences = [seq.get_chain_sequences(x['atoms']) for x in dataset]
    # Flatten.
    flat_chain_sequences = [x for sublist in all_chain_sequences for x in sublist]
    seq.write_to_blast_db(flat_chain_sequences, blast_db_path)

    def filter_fn(df):
        cs = seq.get_chain_sequences(df)
        hits = seq.find_similar(cs, blast_db_path, cutoff, 1)
        if len(hits) > 0:
            # Return empty dataframe.
            return df.iloc[0:0]
        else:
            return df

    return filter_fn