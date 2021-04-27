import argparse
import logging
import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from model import GNN_RSR
from data import GNNTransformRSR
from atom3d.datasets import LMDBDataset
from scipy.stats import spearmanr


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

    targets = []
    decoys = []
    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        batch_losses = F.mse_loss(output, data.y, reduction='none')
        losses.extend(batch_losses.tolist())
        targets.extend(data.target)
        decoys.extend(data.decoy)
        y_true.extend(data.y.tolist())
        y_pred.extend(output.tolist())

    results_df = pd.DataFrame(
        np.array([targets, decoys, y_true, y_pred]).T,
        columns=['target', 'decoy', 'true', 'pred'],
        )

    corrs = compute_correlations(results_df)
    return np.sqrt(np.mean(losses)), corrs, results_df

def plot_corr(y_true, y_pred, plot_dir):
    plt.clf()
    sns.scatterplot(y_true, y_pred)
    plt.xlabel('Actual -log(K)')
    plt.ylabel('Predicted -log(K)')
    plt.savefig(plot_dir)

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'), transform=GNNTransformRSR())
    val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'), transform=GNNTransformRSR())
    test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=GNNTransformRSR())
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)

    for data in train_loader:
        num_features = data.num_features
        break

    model = GNN_RSR(num_features, hidden_dim=args.hidden_dim).to(device)
    model.to(device)

    best_val_loss = 999
    best_rs = 0


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(model, train_loader, optimizer, device)
        val_loss, corrs, results_df = test(model, val_loader, device)
        if corrs['all_spearman'] > best_rs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights.pt'))
            best_rs = corrs['all_spearman']
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Per-target Spearman R: {:.7f}, Global Spearman R: {:.7f}'.format(
            train_loss, val_loss, corrs['per_target_spearman'], corrs['all_spearman']))

    if test_mode:
        test_file = os.path.join(log_dir, f'rsr_rep{rep}.csv')
        model.load_state_dict(torch.load(os.path.join(log_dir, f'best_weights.pt')))
        val_loss, corrs, results_df = test(model, test_loader, device)
        # plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}_test.png'))
        print('\tTest RMSE: {:.7f}, Per-target Spearman R: {:.7f}, Global Spearman R: {:.7f}'.format(
            train_loss, val_loss, corrs['per_target_spearman'], corrs['all_spearman']))
        pd.to_csv(results_df, test_file, index=False)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default=None)
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
            log_dir = os.path.join('logs', f'test_rep{rep}')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train(args, device, log_dir, seed, test_mode=True)
