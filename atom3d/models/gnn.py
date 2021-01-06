import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    """Base graph convolutional neural network architecture for molecular data, using convolutions defined by Kipf and Welling (2017). Each GCN layer is followed by batch normalization and a ReLU nonlinearity. 

    :param in_dim: Dimension of node features
    :type in_dim: int
    :param out_dim: Output dimension
    :type out_dim: int
    :param hidden_dim: Base number of hidden units, defaults to 64
    :type hidden_dim: int, optional
    """    
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*4)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)
        self.conv4 = GCNConv(hidden_dim*4, hidden_dim*4)
        self.bn4 = nn.BatchNorm1d(hidden_dim*4)
        self.conv5 = GCNConv(hidden_dim*4, out_dim)
        self.bn5 = nn.BatchNorm1d(out_dim)


    def forward(self, x, edge_index, edge_weight, batch, select_idx=None):
        """Forward method. The convolutional layers learn an embedding vector for each node in the network, but it is often necessary to reduce this to a single vector for classification or regression. 
        There are two ways to do this: (1) global mean pooling over all nodes (default), or (2) extract the embedding of a single node in the graph supplied by the ``select_idx`` argument.

        Be aware that pytorch-geometric concatenates tensors in each batch (with example membership defined by the ``batch`` tensor), so for batches of size > 1, the selection indices must be modified to account for this. 
        See :func:`atom3d.util.graph.adjust_graph_indices` for an example of how to adjust selection indices in a batch graph.

        The input parameters should be in the format produced by :func:`atom3d.util.graph.prot_df_to_graph` or :func:`atom3d.util.graph.mol_df_to_graph`.

        :param x: Input node features
        :type x: torch.FloatTensor
        :param edge_index: Edges defined in COO format
        :type edge_index: torch.LongTensor
        :param edge_weight: Edge weights
        :type edge_weight: torch.FloatTensor
        :param batch: Batch element membership for each node in batch.
        :type batch: torch.LongTensor
        :param select_idx: Selection indices to specify single node embedding to extract, defaults to None
        :type select_idx: torch.LongTensor, optional
        :return: Output of GNN, with dimension (batch_size, out_dim).
        :rtype: torch.FloatTensor
        """        
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = self.bn5(x)

        # if specified, select embedding of a single node; else sum over nodes
        if select_idx:
            x = torch.index_select(x, 0, select_idx)
        else:
            x = global_mean_pool(x, batch)

        return x