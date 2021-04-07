import argparse
import logging
import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from model import GNN_LBA
from data import GNNTransformLBA
from atom3d.datasets import LMDBDataset
from scipy.stats import spearmanr

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

    r_p = np.corrcoef(y_true, y_pred)[0,1]
    r_s = spearmanr(y_true, y_pred)[0]

    return np.sqrt(loss_all / total), r_p, r_s, y_true, y_pred

def plot_corr(y_true, y_pred, plot_dir):
    plt.clf()
    sns.scatterplot(y_true, y_pred)
    plt.xlabel('Actual -log(K)')
    plt.ylabel('Predicted -log(K)')
    plt.savefig(plot_dir)

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, seed=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'), transform=GNNTransformLBA())
    val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'), transform=GNNTransformLBA())
    test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=GNNTransformLBA())
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)

    for data in train_loader:
        num_features = data.num_features
        break

    model = GNN_LBA(num_features, hidden_dim=args.hidden_dim).to(device)
    model.to(device)

    best_val_loss = 999
    best_rp = 0
    best_rs = 0


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(model, train_loader, optimizer, device)
        val_loss, r_p, r_s, y_true, y_pred = test(model, val_loader, device)
        if val_loss < best_val_loss:
            save_weights(model, os.path.join(log_dir, f'best_weights.pt'))
            # plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}.png'))
            best_val_loss = val_loss
            best_rp = r_p
            best_rs = r_s
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(train_loss, val_loss, r_p, r_s))
        # logger.info('{:03d}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(epoch, train_loss, val_loss, r_p, r_s))

    if test_mode:
        test_file = os.path.join(log_dir, f'test_results.txt')
        model.load_state_dict(torch.load(os.path.join(log_dir, f'best_weights.pt')))
        rmse, pearson, spearman, y_true, y_pred = test(model, test_loader, device)
        # plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}_test.png'))
        print('Test RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(rmse, pearson, spearman))
        with open(test_file, 'a+') as out:
            out.write('{}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(seed, rmse, pearson, spearman))



    return best_val_loss, best_rp, best_rs


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
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
        for seed in np.random.randint(0, 1000, size=3):
            print('seed:', seed)
            log_dir = os.path.join('logs', f'test_{seed}')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train(args, device, log_dir, seed, test_mode=True)