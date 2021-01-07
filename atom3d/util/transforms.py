

def graph_transform(item, keys=['atoms']):
    """Transform for converting dataframes to graphs, to be applied when defining a :mod:`Dataset <atom3d.datasets.datasets>`.
    Operates on Dataset items, assumes that the item contains a "labels" key and all keys specified in ``keys`` argument.

    :param item: Dataset item to transform
    :type item: dict
    :param keys: list of keys to transform, where each key contains a dataset of atoms, defaults to ['atoms']
    :type keys: list, optional
    :return: Transformed Dataset item
    :rtype: dict
    """    
    from torch_geometric.data import Data
    import atom3d.util.graph as gr

    for key in keys:
        node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(item[key])
        item[key] = Data(node_feats, edge_index, edge_feats, y=item['label'], pos=pos)

    return item