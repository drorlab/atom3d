import argparse
import logging
import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from model import GNN_PSR
from data import GNNTransformPSR
from atom3d.datasets import LMDBDataset, PTGDataset

import sys
sys.path.append('..')
from dataset import util as psr_util

def compute_correlations(results):
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

    res = {}
    all_true = results['true'].astype(float)
    all_pred = results['pred'].astype(float)
    res['all_pearson'] = all_true.corr(all_pred, method='pearson')
    res['all_kendall'] = all_true.corr(all_pred, method='kendall')
    res['all_spearman'] = all_true.corr(all_pred, method='spearman')

    res['per_target_pearson'] = per_target['pearson'].mean()
    res['per_target_kendall'] = per_target['kendall'].mean()
    res['per_target_spearman'] = per_target['spearman'].mean()

    print(
        '\nCorrelations (Pearson, Kendall, Spearman)\n'
        '    per-target: ({:.3f}, {:.3f}, {:.3f})\n'
        '    global    : ({:.3f}, {:.3f}, {:.3f})'.format(
        float(res["per_target_pearson"]),
        float(res["per_target_kendall"]),
        float(res["per_target_spearman"]),
        float(res["all_pearson"]),
        float(res["all_kendall"]),
        float(res["all_spearman"])))
    return res

def train_loop(model, loader, optimizer, device):
    model.train()

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

    losses = []
    total = 0

    y_true = []
    y_pred = []
    targets = []
    decoys = []

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
        targets.extend(data.target)
        decoys.extend(data.decoy)
        if it % print_frequency == 0:
            print(f'iter {it}, loss {np.mean(losses)}')

    test_df = pd.DataFrame(
        np.array([targets, decoys, y_true, y_pred]).T,
        columns=['target', 'decoy', 'true', 'pred'],
        )
    
    res = compute_correlations(test_df)

    return np.mean(losses), res, test_df

# def plot_corr(y_true, y_pred, plot_dir):
#     plt.clf()
#     sns.scatterplot(y_true, y_pred)
#     plt.xlabel('Actual -log(K)')
#     plt.ylabel('Predicted -log(K)')
#     plt.savefig(plot_dir)

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    if args.precomputed:
        train_dataset = PTGDataset(os.path.join(args.data_dir, 'train'))
        val_dataset = PTGDataset(os.path.join(args.data_dir, 'val'))
        test_dataset = PTGDataset(os.path.join(args.data_dir, 'test'))

    else:
        train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'), transform=GNNTransformPSR())
        val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'), transform=GNNTransformPSR())
        test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=GNNTransformPSR())
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)

    for data in train_loader:
        num_features = data.num_features
        break

    model = GNN_PSR(num_features, hidden_dim=args.hidden_dim).to(device)
    model.to(device)

    best_val_loss = 999
    best_rp = 0
    best_rs = 0


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                        factor=0.7, patience=3,
    #                                                        min_lr=0.00001)

    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(model, train_loader, optimizer, device)
        val_loss, corrs, results_df = test(model, val_loader, device)
        if val_loss < best_val_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
            best_val_loss = val_loss
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Per-target Spearman R: {:.7f}, Global Spearman R: {:.7f}'.format(
            train_loss, val_loss, corrs['per_target_spearman'], corrs['all_spearman']))

    if test_mode:
        cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
        model.load_state_dict(cpt['model_state_dict'])
        train_file = os.path.join(log_dir, f'psr-rep{rep}.best.train.pt')
        val_file = os.path.join(log_dir, f'psr-rep{rep}.best.val.pt')
        test_file = os.path.join(log_dir, f'psr-rep{rep}.best.test.pt')
        
        _, corrs, results_train = test(model, train_loader, device)
        torch.save(results_train.to_dict('list'), train_file)
        _, corrs, results_val = test(model, val_loader, device)
        torch.save(results_val.to_dict('list'), val_file)
        _, corrs, results_test = test(model, test_loader, device)
        torch.save(results_test.to_dict('list'), test_file)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--precomputed', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir

    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join('logs', now)
        else:
            log_dir = os.path.join('logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train(args, device, log_dir)
        
    elif args.mode == 'test':
        for rep, seed in enumerate(np.random.randint(0, 1000, size=3)):
            print('seed:', seed)
            log_dir = os.path.join('logs', f'psr_test')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train(args, device, log_dir, rep, test_mode=True)
