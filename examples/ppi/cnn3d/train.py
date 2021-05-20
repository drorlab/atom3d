import argparse
import datetime
import json
import os
import time
import tqdm

import numpy as np
import pandas as pd
import sklearn.metrics as sm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from atom3d.datasets import LMDBDataset
from scipy.stats import spearmanr

from model import CNN3D_PPI
from data import CNN3D_Dataset


def major_vote(results):
    data = []
    for ensemble, df in results.groupby('id'):
        true = int(df['true'].unique()[0])
        num_zeros = np.sum(df['pred'] == 0)
        num_ones = np.sum(df['pred'] == 1)
        majority_pred = int(df['pred'].mode().values[0])
        avg_prob = df['prob'].astype(np.float).mean()
        data.append([ensemble, true, majority_pred, avg_prob, num_zeros, num_ones])
    vote_df = pd.DataFrame(data, columns=['id', 'true', 'pred', 'avg_prob',
                                          'num_pred_zeros', 'num_pred_ones'])
    return vote_df


def compute_stats(df):
    results = major_vote(df)
    res = {}
    all_true = results['true'].astype(np.int8)
    all_pred = results['pred'].astype(np.int8)
    res['auroc'] = sm.roc_auc_score(all_true, all_pred)
    res['auprc'] = sm.average_precision_score(all_true, all_pred)
    res['acc'] = sm.accuracy_score(all_true, all_pred.round())
    res['bal_acc'] = \
        sm.balanced_accuracy_score(all_true, all_pred.round())
    return res


# Construct model
def conv_model(in_channels, spatial_size, args):
    num_conv = args.num_conv
    conv_filters = [32 * (2**n) for n in range(num_conv)]
    conv_kernel_size = 3
    max_pool_positions = [0, 1]*int((num_conv+1)/2)
    max_pool_sizes = [2]*num_conv
    max_pool_strides = [2]*num_conv
    fc_units = [512]
    top_fc_units = [512]*args.num_final_fc_layers

    model = CNN3D_PPI(
        in_channels, spatial_size,
        args.conv_drop_rate,
        args.fc_drop_rate,
        args.top_nn_drop_rate,
        conv_filters, conv_kernel_size,
        max_pool_positions,
        max_pool_sizes, max_pool_strides,
        fc_units,
        top_fc_units,
        batch_norm=args.batch_norm,
        dropout=not args.no_dropout)
    return model


def train_loop(model, loader, criterion, optimizer, device):
    model.train()

    loss_all = 0
    total = 0
    progress_format = 'train loss: {:6.6f}'
    with tqdm.tqdm(total=len(loader), desc=progress_format.format(0)) as t:
        for i, data in enumerate(loader):
            feature_left = data['feature_left'].to(device).to(torch.float32)
            feature_right = data['feature_right'].to(device).to(torch.float32)
            labels = data['label'].to(device).to(torch.float32)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = model(feature_left, feature_right)
            loss = criterion(output, labels)
            loss.backward()
            loss_all += loss.item() * len(labels)
            total += len(labels)
            optimizer.step()
            # stats
            t.set_description(progress_format.format(np.sqrt(loss_all/total)))
            t.update(1)

    return np.sqrt(loss_all / total)


@torch.no_grad()
def test(model, loader, criterion, device):
    model.eval()

    loss_all = 0
    total = 0

    ids = []
    y_true = []
    y_probs = []
    y_pred = []

    for data in loader:
        feature_left = data['feature_left'].to(device).to(torch.float32)
        feature_right = data['feature_right'].to(device).to(torch.float32)
        labels = data['label'].to(device).to(torch.float32)
        output = model(feature_left, feature_right)

        loss = criterion(output, labels)
        loss_all += loss.item() * len(labels)
        total += len(labels)

        ids.extend(data['id'])
        y_true.extend(labels.int().tolist())
        y_probs.extend(output.tolist())

        preds = torch.round(output).int()
        y_pred.extend(preds.tolist())

    results_df = pd.DataFrame(
        np.array([ids, y_true, y_probs, y_pred]).T,
        columns=['id', 'true', 'prob', 'pred'],
        )

    stats = compute_stats(results_df)
    return np.sqrt(loss_all / total), stats, results_df


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def train(args, device, test_mode=False):
    print("Training model with config:")
    print(str(json.dumps(args.__dict__, indent=4)) + "\n")

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    train_dataset = CNN3D_Dataset(
        os.path.join(args.data_dir, 'train'),
        testing=False,
        random_seed=args.random_seed,
        )
    val_dataset = CNN3D_Dataset(
        os.path.join(args.data_dir, 'val'),
        testing=False,
        random_seed=args.random_seed,
        )
    test_dataset = CNN3D_Dataset(
        os.path.join(args.data_dir, 'test'),
        testing=True,
        random_seed=args.random_seed,
        )

    train_loader = DataLoader(train_dataset, args.batch_size)
    val_loader = DataLoader(val_dataset, args.batch_size)
    test_loader = DataLoader(test_dataset, args.batch_size)

    for data in train_loader:
        in_channels, spatial_size = data['feature_left'].size()[1:3]
        print('num channels: {:}, spatial size: {:}'.format(in_channels, spatial_size))
        break

    model = conv_model(in_channels, spatial_size, args)
    print(model)
    model.to(device)

    prev_val_loss = np.Inf
    best_val_loss = np.Inf
    best_val_auroc = 0
    best_stats = None

    criterion = nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(model, train_loader, criterion, optimizer, device)
        val_loss, stats, val_df = test(model, val_loader, criterion, device)
        elapsed = (time.time() - start)
        print(f'Epoch {epoch:03d} finished in : {elapsed:.3f} s')
        print(f"\tTrain loss {train_loss:.4f}, Val loss: {val_loss:.4f}, "
              f"Val AUROC: {stats['auroc']:.4f}, Val AUPRC: {stats['auroc']:.4f}")
        #if stats['auroc'] > best_val_auroc:
        if val_loss < best_val_loss:
            print(f"\nSave model at epoch {epoch:03d}, val_loss: {val_loss:.4f}, "
                  f"auroc: {stats['auroc']:.4f}, auprc: {stats['auroc']:.4f}")
            save_weights(model, os.path.join(args.output_dir, f'best_weights.pt'))
            best_val_loss = val_loss
            best_val_auroc = stats['auroc']
            best_stats = stats
        if args.early_stopping and val_loss >= prev_val_loss:
            print(f"Validation loss stopped decreasing, stopping at epoch {epoch:03d}...")
            break
        prev_val_loss = val_loss

    if test_mode:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_weights.pt')))
        test_loss, stats, test_df = test(model, test_loader, criterion, device)
        test_df.to_pickle(os.path.join(args.output_dir, 'test_results.pkl'))
        print(f"Test loss: {test_loss:.4f}, Test AUROC: {stats['auroc']:.4f}, "
              f"Test AUPRC: {stats['auprc']:.4f}")
        test_file = os.path.join(args.output_dir, f'test_results.txt')
        with open(test_file, 'w') as f:
            f.write(f"test_loss\tAUROC\tAUPRC\n")
            f.write(f"{test_loss:}\t{stats['auroc']:}\t{stats['auprc']:}\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.environ['PPI_DATA'])
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test'])
    parser.add_argument('--output_dir', type=str, default=os.environ['LOG_DIR'])
    parser.add_argument('--unobserved', action='store_true', default=False)

    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--conv_drop_rate', type=float, default=0.1)
    parser.add_argument('--fc_drop_rate', type=float, default=0.25)
    parser.add_argument('--top_nn_drop_rate', type=float, default=0.25)
    parser.add_argument('--num_epochs', type=int, default=30)

    parser.add_argument('--num_conv', type=int, default=4)
    parser.add_argument('--num_final_fc_layers', type=int, default=2)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--no_dropout', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=int(np.random.randint(1, 10e6)))

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up output dir
    args.output_dir = os.path.join(args.output_dir, 'ppi')
    assert args.output_dir != None
    if args.unobserved:
        args.output_dir = os.path.join(args.output_dir, 'None')
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        num = 0
        while True:
            dirpath = os.path.join(args.output_dir, str(num))
            if os.path.exists(dirpath):
                num += 1
            else:
                args.output_dir = dirpath
                print('Creating output directory {:}'.format(args.output_dir))
                os.makedirs(args.output_dir)
                break

    print(f"Running mode {args.mode:} with seed {args.random_seed:} "
          f"and output dir {args.output_dir}")
    train(args, device, args.mode=='test')
