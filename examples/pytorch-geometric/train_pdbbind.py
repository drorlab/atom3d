#
# GCN torch-geometric training script for PDBBind
# based on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py

import os
import time
import logging
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import argparse

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, MSELoss
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_add_pool

from pdbbind_dataloader import pdbbind_dataloader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv5 = GCNConv(hidden_dim, hidden_dim)
        self.bn5 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, 1)


    def forward(self, x, edge_index, edge_weight, batch):
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
        x = global_add_pool(x, batch)
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        return self.fc2(x).view(-1)

class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GIN, self).__init__()

        nn1 = Sequential(Linear(num_features, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        nn2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        nn3 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)

        nn4 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim)

        nn5 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(hidden_dim)

        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x).view(-1)



def train(epoch, arch, model, loader, optimizer, device):
    model.train()

    # if epoch == 51:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    total = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        if arch == 'GCN':
            output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        elif arch == 'GIN':
            output = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        optimizer.step()
    return np.sqrt(loss_all / total)


@torch.no_grad()
def test(arch, model, loader, device):
    model.eval()

    loss_all = 0
    total = 0

    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        if arch == 'GCN':
            output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        elif arch == 'GIN':
            output = model(data.x, data.edge_index, data.batch)
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
    sns.scatterplot(y_true, y_pred)
    plt.xlabel('Actual -log(K)')
    plt.ylabel('Predicted -log(K)')
    plt.savefig(plot_dir)

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def train_pdbbind(split, architecture, base_dir, device, log_dir, seed=None, test_mode=False):
    logger = logging.getLogger('pdbbind_log')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    num_epochs = 100
    batch_size = 32
    hidden_dim = 64
    learning_rate = 1e-4
    split_dir = os.path.join(os.getcwd(), base_dir, 'splits')
    train_split = os.path.join(split_dir, 'core_split', f'train_{split}.txt')
    val_split = os.path.join(split_dir, 'core_split', f'val_{split}.txt')
    test_split = os.path.join(split_dir, 'core_split', f'test.txt')
    train_loader = pdbbind_dataloader(batch_size, split_file=train_split)
    val_loader = pdbbind_dataloader(batch_size, split_file=val_split)
    test_loader = pdbbind_dataloader(batch_size, split_file=test_split)

    if not os.path.exists(os.path.join(log_dir, 'params.txt')):
        with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
            f.write(f'Split method: {split}\n')
            f.write(f'Model: {architecture}\n')
            f.write(f'Epochs: {num_epochs}\n')
            f.write(f'Batch size: {batch_size}\n')
            f.write(f'Hidden dim: {hidden_dim}\n')
            f.write(f'Learning rate: {learning_rate}')

    for data in train_loader:
        num_features = data.num_features
        break

    if architecture == 'GCN':
        model = GCN(num_features, hidden_dim=hidden_dim).to(device)
    elif architecture == 'GIN':
        model = GIN(num_features, hidden_dim=hidden_dim).to(device) 

    best_val_loss = 999
    best_rp = 0
    best_rs = 0


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_loss = train(epoch, architecture, model, train_loader, optimizer, device)
        val_loss, r_p, r_s, y_true, y_pred = test(architecture, model, val_loader, device)
        if val_loss < best_val_loss:
            save_weights(model, os.path.join(log_dir, f'best_weights_{split}.pt'))
            plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}.png'))
            best_val_loss = val_loss
            best_rp = r_p
            best_rs = r_s
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(train_loss, val_loss, r_p, r_s))
        logger.info('{:03d}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(epoch, train_loss, val_loss, r_p, r_s))

    if test:
        test_file = os.path.join(log_dir, f'test_results_{split}.txt')
        model.load_state_dict(torch.load(os.path.join(log_dir, f'best_weights_{split}.pt')))
        rmse, pearson, spearman, y_true, y_pred = test(architecture, model, test_loader, device)
        plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}_test.png'))
        print('Test RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(rmse, pearson, spearman))
        with open(test_file, 'a+') as out:
            out.write('{}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(seed, rmse, pearson, spearman))



    return best_val_loss, best_rp, best_rs


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--split', type=str, default='random')
    parser.add_argument('--architecture', type=str, default='GCN')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = '../../data/pdbbind'
    log_dir = args.log_dir


    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(base_dir, 'logs', now)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        train_pdbbind(args.split, args.architecture, base_dir, device, log_dir)
    elif args.mode == 'test':
        for seed in np.random.randint(0, 1000, size=3):
            print('seed:', seed)
            log_dir = os.path.join(base_dir, 'logs', f'test_{args.split}_{seed}')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_pdbbind(args.split, args.architecture, base_dir, device, log_dir, seed, test_mode=True)







