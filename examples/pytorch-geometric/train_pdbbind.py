#
# GCN torch-geometric training script for PDBBind
# based on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py

import os
import time
import logging
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, MSELoss
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool

from pdbbind_dataloader import pdbbind_dataloader

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Net(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = Linear(hidden_dim, 1)


    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn1(x)
        # x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn2(x)
        # x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = global_add_pool(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        return self.fc(x).view(-1)


def train(epoch, model, loader, optimizer, device):
    model.train()

    # if epoch == 51:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    total = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        optimizer.step()
    return np.sqrt(loss_all / total)


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    loss_all = 0
    total = 0

    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        loss = F.mse_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        y_true.extend(data.y.tolist())
        y_pred.extend(output.tolist())


    # vx = np.array(y_pred) - np.mean(y_pred)
    # vy = np.array(y_true) - np.mean(y_true)
    # r2 = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    r_p = np.corrcoef(y_true, y_pred)[0,1]
    r_s = spearmanr(y_true, y_pred)[0]

    # r2 = r2_score(y_true, y_pred)
    return np.sqrt(loss_all / total), r_p, r_s, y_true, y_pred

def plot_corr(y_true, y_pred, plot_dir):
    plt.clf()
    plt.scatter(y_true, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(plot_dir)

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def main(fold=0, split='random', base_dir='../../data/pdbbind/', log_dir='../../data/pdbbind/logs'):
    logger = logging.getLogger('pdbbind_log')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    num_epochs = 100
    batch_size = 32
    hidden_dim = 32
    learning_rate = 1e-4
    split_dir = os.path.join(os.getcwd(), base_dir, 'splits')
    train_split = os.path.join(split_dir, 'core_split', f'train_{split}_cv{fold}.txt')
    val_split = os.path.join(split_dir, 'core_split', f'val_{split}_cv{fold}.txt')
    train_loader = pdbbind_dataloader(batch_size, split_file=train_split)
    val_loader = pdbbind_dataloader(batch_size, split_file=val_split)

    if not os.path.exists(os.path.join(log_dir, 'params.txt')):
        with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
            f.write(f'Split method: {split}')
            f.write(f'Epochs: {num_epochs}')
            f.write(f'Batch size: {batch_size}')
            f.write(f'Hidden dim: {hidden_dim}')
            f.write(f'Learning rate: {learning_rate}')

    best_val_loss = 999
    best_rp = 0
    best_rs = 0

    for data in train_loader:
        num_features = data.num_features
        break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_features, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_loss = train(epoch, model, train_loader, optimizer, device)
        val_loss, r_p, r_s, y_true, y_pred = test(model, val_loader, device)
        if val_loss < best_val_loss:
            save_weights(model, os.path.join(log_dir, f'best_weights_{split}_fold{fold}.pt'))
            plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}_cv{fold}.png'))
            best_val_loss = val_loss
            best_rp = r_p
            best_rs = r_s
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(train_loss, val_loss, r_p, r_s))
        logger.info('{}\t{:03d}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(fold, epoch, train_loss, val_loss, r_p, r_s))

    return best_val_loss, best_rp, best_rs


if __name__=="__main__":
    main()
