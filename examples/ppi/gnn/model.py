import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

class GNN_PPI(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GNN_PPI, self).__init__()
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

    def forward(self, x, edge_index, edge_weight, ca_idx, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = self.bn5(x)
        x = torch.index_select(x, 0, ca_idx)
        return x
    
class MLP_PPI(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(MLP_PPI, self).__init__()
        self.fc1 = nn.Linear(hidden_dim*4, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, 1)

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x).view(-1)