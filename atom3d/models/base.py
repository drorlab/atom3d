import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import Linear


class GCN(nn.Module):
    """Base graph convolutional neural network architecture for molecular data, using convolutions defined by Kipf and Welling (2017).

    :param num_features: Dimension of node features.
    :type num_features: int
    :param hidden_dim: Base number of hidden units, defaults to 64
    :type hidden_dim: int, optional
    """    
    def __init__(self, num_features, hidden_dim=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*4)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)
        self.conv4 = GCNConv(hidden_dim*4, hidden_dim*4)
        self.bn4 = nn.BatchNorm1d(hidden_dim*4)
        self.conv5 = GCNConv(hidden_dim*4, hidden_dim*2)
        self.bn5 = nn.BatchNorm1d(hidden_dim*2)
        self.fc1 = Linear(hidden_dim*2, hidden_dim*2)
        self.fc2 = Linear(hidden_dim*2, 20)


    def forward(self, x, edge_index, edge_weight, batch, select_idx=None):
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
            x = global_add_pool(x, batch)

        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        return self.fc2(x)


class CNN3D(nn.Module):
    """Base 3D convolutional neural network architecture for molecular data.

    :param in_dim: Input dimension.
    :type in_dim: int
    :param out_dim: Output dimension.
    :type out_dim: int
    :param hidden_dim: Base number of hidden units, defaults to 64
    :type hidden_dim: int, optional
    """    
    def __init__(self, in_channels, out_channels=20, hidden_dim=64):
        super(CNN3D, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Conv3d(hidden_dim, hidden_dim * 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            
            nn.Conv3d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Conv3d(hidden_dim * 4, hidden_dim * 8, 3, 1, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Conv3d(hidden_dim * 8, hidden_dim * 16, 3, 1, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Conv3d(hidden_dim * 16, out_channels, 5, 1, 0, bias=False),
        )


    def forward(self, input):
        bs = input.size()[0]

        output = self.model(input)
        return output.view(bs, -1)
