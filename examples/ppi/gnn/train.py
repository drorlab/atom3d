import argparse
import logging
import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import GNN_PPI, MLP_PPI
from data import PPIDataset, CollaterPPI
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

def train_loop(epoch, gcn_model, ff_model, loader, criterion, optimizer, device):
    gcn_model.train()
    ff_model.train()

    start = time.time()

    losses = []
    total = 0
    print_frequency = 10
    for it, (graph1, graph2) in enumerate(loader):
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        optimizer.zero_grad()
        out_graph1 = gcn_model(graph1.x, graph1.edge_index, graph1.edge_attr.view(-1), graph1.ca_idx, graph1.batch)
        out_graph2 = gcn_model(graph2.x, graph2.edge_index, graph2.edge_attr.view(-1), graph2.ca_idx, graph2.batch)
        output = ff_model(out_graph1, out_graph2)
        loss = criterion(output, graph1.y)
        loss.backward()
        # loss_all += loss.item() * graph1.num_graphs
        # total += graph1.num_graphs
        losses.append(loss.item())
        optimizer.step()

        if it % print_frequency == 0:
            elapsed = time.time() - start
            print(f'Epoch {epoch}, iter {it}, train loss {np.mean(losses)}, avg it/sec {print_frequency / elapsed}')
            start = time.time()

    return np.mean(losses)


@torch.no_grad()
def test(gcn_model, ff_model, loader, criterion, device):
    gcn_model.eval()
    ff_model.eval()

    losses = []
    total = 0
    print_frequency = 10

    y_true = []
    y_pred = []

    for it, (graph1, graph2) in enumerate(loader):
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        out_graph1 = gcn_model(graph1.x, graph1.edge_index, graph1.edge_attr.view(-1), graph1.ca_idx, graph1.batch)
        out_graph2 = gcn_model(graph2.x, graph2.edge_index, graph2.edge_attr.view(-1), graph2.ca_idx, graph2.batch)
        output = ff_model(out_graph1, out_graph2)
        loss = criterion(output, graph1.y)
        # loss_all += loss.item() * graph1.num_graphs
        # total += graph1.num_graphs
        losses.append(loss.item())
        y_true.extend(graph1.y.tolist())
        y_pred.extend(output.tolist())
        if it % print_frequency == 0:
            print(f'iter {it}, loss {np.mean(losses)}')

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    return np.mean(losses), auroc, auprc

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, seed=None, test_mode=False):
    train_dataset = PPIDataset(os.path.join('/scratch/users/raphtown/atom3d_mirror/lmdb/PPI/splits/DIPS-split/data', 'train'))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=CollaterPPI(batch_size=args.batch_size), num_workers=4)
    val_dataset = PPIDataset(os.path.join('/scratch/users/raphtown/atom3d_mirror/lmdb/PPI/splits/DIPS-split/data', 'val'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=CollaterPPI(batch_size=args.batch_size), num_workers=4)
    test_dataset = PPIDataset('/scratch/users/raphtown/atom3d_mirror/lmdb/PPI/raw/DB5/data')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=CollaterPPI(batch_size=args.batch_size), num_workers=4)
    
    for graph1, graph2 in train_loader:
        print(graph1.num_features)
        num_features = graph1.num_features
        break

    gcn_model = GNN_PPI(num_features, hidden_dim=args.hidden_dim).to(device)
    gcn_model.to(device)
    ff_model = MLP_PPI(args.hidden_dim).to(device)

    best_val_loss = 999
    best_val_auroc = 0


    params = [x for x in gcn_model.parameters()] + [x for x in ff_model.parameters()]

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]))
    criterion.to(device)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(epoch, gcn_model, ff_model, train_loader, criterion, optimizer, device)
        print('validating...')
        val_loss, auroc, auprc = test(gcn_model, ff_model, val_loader, criterion, device)
        if auroc > best_val_auroc:
            torch.save({
                'epoch': epoch,
                'gcn_state_dict': gcn_model.state_dict(),
                'ff_state_dict': ff_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights.pt'))
            best_val_auroc = auroc
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(f'\tTrain loss {train_loss}, Val loss {val_loss}, Val AUROC {auroc}, Val auprc {auprc}')

    if test_mode:
        test_file = os.path.join(log_dir, f'test_results.txt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights.pt'))
        gcn_model.load_state_dict(cpt['gcn_state_dict'])
        ff_model.load_state_dict(cpt['ff_state_dict'])
        test_loss, auroc, auprc = test(gcn_model, ff_model, test_loader, criterion, device)
        print(f'\tTest loss {test_loss}, Test AUROC {auroc}, Test auprc {auprc}')
        with open(test_file, 'w') as f:
            f.write(f'test_loss\tAUROC\n')
            f.write(f'{test_loss}\t{auroc}\n')
        return test_loss, auroc, auprc



    return best_val_loss


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
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