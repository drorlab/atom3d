#
# GCN torch-geometric training script for psp
# based on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py

import argparse
import datetime
import logging
import os
import time

import atom3d.psp.util as psp_util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psp_dataloader as dl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*4)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)
        self.conv4 = GCNConv(hidden_dim*4, hidden_dim*4)
        self.bn4 = nn.BatchNorm1d(hidden_dim*4)
        self.conv5 = GCNConv(hidden_dim*4, hidden_dim*4)
        self.bn5 = nn.BatchNorm1d(hidden_dim*4)
        self.fc1 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, 1)


    def forward(self, x, edge_index, edge_weight, batch):
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
        x = global_mean_pool(x, batch)
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        return self.fc2(x).view(-1)

def compute_global_correlations(results):
    per_target = []
    for key, val in results.groupby(['target']):
        # Ignore target with 2 decoys only since the correlations are
        # not really meaningful.
        if val.shape[0] < 3:
            continue
        true = val['true'].astype(float)
        pred = val['pred'].astype(float)
        pearson = true.corr(pred, method='pearson')
        kendall = true.corr(pred, method='kendall')
        spearman = true.corr(pred, method='spearman')
        per_target.append((key, pearson, kendall, spearman))
    per_target = pd.DataFrame(
        data=per_target,
        columns=['target', 'pearson', 'kendall', 'spearman'])

    # Save metrics.
    res = {}
    all_true = results['true'].astype(float)
    all_pred = results['pred'].astype(float)
    res['all_pearson'] = all_true.corr(all_pred, method='pearson')
    res['all_kendall'] = all_true.corr(all_pred, method='kendall')
    res['all_spearman'] = all_true.corr(all_pred, method='spearman')

    res['per_target_mean_pearson'] = per_target['pearson'].mean()
    res['per_target_mean_kendall'] = per_target['kendall'].mean()
    res['per_target_mean_spearman'] = per_target['spearman'].mean()

    res['per_target_median_pearson'] = per_target['pearson'].median()
    res['per_target_median_kendall'] = per_target['kendall'].median()
    res['per_target_median_spearman'] = per_target['spearman'].median()
    return res

def train(epoch, model, loader, optimizer, device):
    model.train()

    start = time.time()

    losses = []
    total = 0
    print_frequency = 100

    for it, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        losses.append(loss.item())
        # loss_all += loss.item() * data.num_graphs
        # total += data.num_graphs
        optimizer.step()

        if it % print_frequency == 0:
            elapsed = time.time() - start
            print(f'Epoch {epoch}, iter {it}, train loss {np.mean(losses)}, avg it/sec {print_frequency / elapsed}')
            start = time.time()

    return np.mean(losses)


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    losses = []
    total = 0

    y_true = []
    y_pred = []
    structs = []

    print_frequency = 10

    for it, data in enumerate(loader):
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        loss = F.mse_loss(output, data.y)
        losses.append(loss.item())
        # loss_all += loss.item() * data.num_graphs
        # total += data.num_graphs
        y_true.extend([x.item() for x in data.y])
        y_pred.extend(output.tolist())
        structs.extend(data.name)
        if it % print_frequency == 0:
            print(f'iter {it}, loss {np.mean(losses)}')

    test_df = pd.DataFrame(
        np.array([structs, y_true, y_pred]).T,
        columns=['structure', 'true', 'pred'],
        )

    test_df['target'] = test_df.structure.apply(
        lambda x: psp_util.get_target_name(x))
    
    res = compute_global_correlations(test_df)

    # r2 = r2_score(y_true, y_pred)
    return np.mean(losses), res, test_df

def plot_corr(y_true, y_pred, plot_dir):
    plt.clf()
    sns.scatterplot(y_true, y_pred)
    plt.xlabel('Actual -log(K)')
    plt.ylabel('Predicted -log(K)')
    plt.savefig(plot_dir)



def train_psp(data_dir, device, log_dir, checkpoint, seed=None, test_mode=False):
    logger = logging.getLogger('psp_log')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    num_epochs = 20
    batch_size = 32
    hidden_dim = 64
    learning_rate = 1e-4
    

    # train_set = dl.PSP_Dataset(os.path.join(data_dir, 'split_hdf/decoy_50/train_decoy_50@508'), os.path.join(data_dir, 'labels/scores'))
    # train_loader = dl.DataLoader(train_set, batch_size=batch_size, num_workers=8)
    # val_set = dl.PSP_Dataset(os.path.join(data_dir, 'split_hdf/decoy_50/val_decoy_50@56'), os.path.join(data_dir, 'labels/scores'))
    # val_loader = dl.DataLoader(val_set, batch_size=batch_size, num_workers=8)

    train_set = dl.PSP_Dataset_PTG(os.path.join(data_dir, 'train'))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_set = dl.PSP_Dataset_PTG(os.path.join(data_dir, 'val'))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8)

    if not os.path.exists(os.path.join(log_dir, 'params.txt')):
        with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
            f.write(f'Epochs: {num_epochs}\n')
            f.write(f'Batch size: {batch_size}\n')
            f.write(f'Hidden dim: {hidden_dim}\n')
            f.write(f'Learning rate: {learning_rate}')

    for data in train_loader:
        num_features = data.num_features
        break

    model = GCN(num_features, hidden_dim=hidden_dim).to(device)
    model.to(device)

    best_val_loss = 999
    best_val_corr = 0


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=3,
                                                           min_lr=0.00001)

    if checkpoint:
        cpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(cpt['model_state_dict'])
        optimizer.load_state_dict(cpt['optimizer_state_dict'])

    
    print(f'training for {num_epochs} epochs')
    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_loss = train(epoch, model, train_loader, optimizer, device)
        print('validating...')
        val_loss, res, test_df = test(model, val_loader, device)
        scheduler.step(val_loss)
        if res['all_spearman'] > best_val_corr:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights.pt'))
            best_val_corr = res['all_spearman']
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(
            '\nVal Correlations (Pearson, Kendall, Spearman)\n'
            '    per-target averaged median: ({:.3f}, {:.3f}, {:.3f})\n'
            '    per-target averaged mean: ({:.3f}, {:.3f}, {:.3f})\n'
            '    all averaged: ({:.3f}, {:.3f}, {:.3f})'.format(
            float(res["per_target_median_pearson"]),
            float(res["per_target_median_kendall"]),
            float(res["per_target_median_spearman"]),
            float(res["per_target_mean_pearson"]),
            float(res["per_target_mean_kendall"]),
            float(res["per_target_mean_spearman"]),
            float(res["all_pearson"]),
            float(res["all_kendall"]),
            float(res["all_spearman"])))
        # print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}, Kendall Tau: {:.7f}'.format(train_loss, val_loss, r_p, r_s, r_k))
        # logger.info('{:03d}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(epoch, train_loss, val_loss, r_p, r_s))
    
    if test_mode:
        print('testing...')
        # test_set = dl.PSP_Dataset(os.path.join(data_dir, 'split_hdf/decoy_50/test_decoy_all@85'), os.path.join(data_dir, 'labels/scores'))
        # test_loader = dl.DataLoader(test_set, batch_size=batch_size, num_workers=8)
        test_set = dl.PSP_Dataset_PTG(os.path.join(data_dir, 'test'))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8)
        test_file = os.path.join(log_dir, f'test_results.txt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights.pt'))
        model.load_state_dict(cpt['model_state_dict'])
        rmse, res, test_df = test(model, test_loader, device)
        print(
            '\nTest Correlations (Pearson, Kendall, Spearman)\n'
            '    per-target averaged median: ({:.3f}, {:.3f}, {:.3f})\n'
            '    per-target averaged mean: ({:.3f}, {:.3f}, {:.3f})\n'
            '    all averaged: ({:.3f}, {:.3f}, {:.3f})'.format(
            float(res["per_target_median_pearson"]),
            float(res["per_target_median_kendall"]),
            float(res["per_target_median_spearman"]),
            float(res["per_target_mean_pearson"]),
            float(res["per_target_mean_kendall"]),
            float(res["per_target_mean_spearman"]),
            float(res["all_pearson"]),
            float(res["all_kendall"]),
            float(res["all_spearman"])))
        test_df.to_csv(test_file)

    return best_val_loss


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = '../../data/psp'
    data_dir = '../../data/psp/graph_pt'
    log_dir = args.log_dir


    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(base_dir, 'logs', now)
        else:
            log_dir = os.path.join(base_dir, 'logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train_psp(data_dir, device, log_dir, args.checkpoint)
    elif args.mode == 'test':
        for seed in np.random.randint(0, 1000, size=3):
            print('seed:', seed)
            log_dir = os.path.join(base_dir, 'logs', f'test_{seed}')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_psp(data_dir, device, log_dir, args.checkpoint, seed, test_mode=True)








