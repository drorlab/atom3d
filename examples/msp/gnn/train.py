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
from model import GNN_MSP, MLP_MSP
from data import GNNTransformMSP, CollaterMSP
from atom3d.datasets import LMDBDataset, PTGDataset
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

def train_loop(epoch, gcn_model, ff_model, loader, criterion, optimizer, device):
    gcn_model.train()
    ff_model.train()

    start = time.time()

    losses = []
    total = 0
    print_frequency = 10
    for it, (original, mutated) in enumerate(loader):
        original = original.to(device)
        mutated = mutated.to(device)
        optimizer.zero_grad()
        out_original = gcn_model(original.x, original.edge_index, original.edge_attr.view(-1), original.mut_idx, original.batch)
        out_mutated = gcn_model(mutated.x, mutated.edge_index, mutated.edge_attr.view(-1), mutated.mut_idx, mutated.batch)
        output = ff_model(out_original, out_mutated)
        loss = criterion(output, original.y)
        loss.backward()
        # loss_all += loss.item() * original.num_graphs
        # total += original.num_graphs
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

    for it, (original, mutated) in enumerate(loader):
        original = original.to(device)
        mutated = mutated.to(device)
        out_original = gcn_model(original.x, original.edge_index, original.edge_attr.view(-1), original.mut_idx, original.batch)
        out_mutated = gcn_model(mutated.x, mutated.edge_index, mutated.edge_attr.view(-1), mutated.mut_idx, mutated.batch)
        output = ff_model(out_original, out_mutated)
        loss = criterion(output, original.y)
        # loss_all += loss.item() * original.num_graphs
        # total += original.num_graphs
        losses.append(loss.item())
        y_true.extend(original.y.tolist())
        y_pred.extend(torch.sigmoid(output).tolist())
        # if it % print_frequency == 0:
        #     print(f'iter {it}, loss {np.mean(losses)}')

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    return np.mean(losses), auroc, auprc, y_true, y_pred


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)
    transform = GNNTransformMSP()
    if args.precomputed:
        train_dataset = PTGDataset(os.path.join(args.data_dir, 'train'))
        val_dataset = PTGDataset(os.path.join(args.data_dir, 'val'))
        test_dataset = PTGDataset(os.path.join(args.data_dir, 'test'))
    else:
        train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'), transform=transform)
        val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'), transform=transform)
        test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, collate_fn=CollaterMSP(batch_size=args.batch_size))
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=CollaterMSP(batch_size=args.batch_size))
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=CollaterMSP(batch_size=args.batch_size))
    
    for original, mutated in train_loader:
        num_features = original.num_features
        break

    gcn_model = GNN_MSP(num_features, hidden_dim=args.hidden_dim).to(device)
    gcn_model.to(device)
    ff_model = MLP_MSP(args.hidden_dim).to(device)

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
        val_loss, auroc, auprc, _, _ = test(gcn_model, ff_model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            torch.save({
                'epoch': epoch,
                'gcn_state_dict': gcn_model.state_dict(),
                'ff_state_dict': ff_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
            best_val_loss = val_loss
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(f'\tTrain loss {train_loss}, Val loss {val_loss}, Val AUROC {auroc}, Val auprc {auprc}')

    if test_mode:
        train_file = os.path.join(log_dir, f'msp-rep{rep}.best.train.pt')
        val_file = os.path.join(log_dir, f'msp-rep{rep}.best.val.pt')
        test_file = os.path.join(log_dir, f'msp-rep{rep}.best.test.pt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
        gcn_model.load_state_dict(cpt['gcn_state_dict'])
        ff_model.load_state_dict(cpt['ff_state_dict'])
        _, _, _, y_true_train, y_pred_train = test(gcn_model, ff_model, train_loader, criterion, device)
        torch.save({'targets':y_true_train, 'predictions':y_pred_train}, train_file)
        _, _, _, y_true_val, y_pred_val = test(gcn_model, ff_model, val_loader, criterion, device)
        torch.save({'targets':y_true_val, 'predictions':y_pred_val}, val_file)
        test_loss, auroc, auprc, y_true_test, y_pred_test = test(gcn_model, ff_model, test_loader, criterion, device)
        print(f'\tTest loss {test_loss}, Test AUROC {auroc}, Test auprc {auprc}')
        torch.save({'targets':y_true_test, 'predictions':y_pred_test}, test_file)
        return test_loss, auroc, auprc



    return best_val_loss


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
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
            log_dir = os.path.join('logs', f'msp_test')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train(args, device, log_dir, rep, test_mode=True)