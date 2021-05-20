import argparse
import datetime
import json
import os
import time
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from atom3d.datasets import LMDBDataset
from scipy.stats import spearmanr

from model import CNN3D_PSR
from data import CNN3D_TransformPSR


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


# Construct model
def conv_model(in_channels, spatial_size, args):
    num_conv = args.num_conv
    conv_filters = [32 * (2**n) for n in range(num_conv)]
    conv_kernel_size = 3
    max_pool_positions = [0, 1]*int((num_conv+1)/2)
    max_pool_sizes = [2]*num_conv
    max_pool_strides = [2]*num_conv
    fc_units = [512]

    model = CNN3D_PSR(
        in_channels, spatial_size,
        args.conv_drop_rate,
        args.fc_drop_rate,
        conv_filters, conv_kernel_size,
        max_pool_positions,
        max_pool_sizes, max_pool_strides,
        fc_units,
        batch_norm=args.batch_norm,
        dropout=not args.no_dropout)
    return model


def train_loop(model, loader, optimizer, device):
    model.train()

    loss_all = 0
    total = 0
    progress_format = 'train loss: {:6.6f}'
    with tqdm.tqdm(total=len(loader), desc=progress_format.format(0)) as t:
        for i, data in enumerate(loader):
            feature = data['feature'].to(device).to(torch.float32)
            label = data['label'].to(device).to(torch.float32)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = model(feature)
            loss = F.mse_loss(output, label)
            loss.backward()
            loss_all += loss.item() * len(label)
            total += len(label)
            optimizer.step()
            # stats
            t.set_description(progress_format.format(np.sqrt(loss_all/total)))
            t.update(1)

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
        feature = data['feature'].to(device).to(torch.float32)
        label = data['label'].to(device).to(torch.float32)
        output = model(feature)
        batch_losses = F.mse_loss(output, label, reduction='none')
        losses.extend(batch_losses.tolist())
        targets.extend(data['target'])
        decoys.extend(data['decoy'])
        y_true.extend(label.tolist())
        y_pred.extend(output.tolist())

    results_df = pd.DataFrame(
        np.array([targets, decoys, y_true, y_pred]).T,
        columns=['target', 'decoy', 'true', 'pred'],
        )

    corrs = compute_correlations(results_df)
    return np.sqrt(np.mean(losses)), corrs, results_df


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

    train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'),
                                transform=CNN3D_TransformPSR(random_seed=args.random_seed))
    val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'),
                              transform=CNN3D_TransformPSR(random_seed=args.random_seed))
    test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'),
                               transform=CNN3D_TransformPSR(random_seed=args.random_seed))

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    for data in train_loader:
        in_channels, spatial_size = data['feature'].size()[1:3]
        print('num channels: {:}, spatial size: {:}'.format(in_channels, spatial_size))
        break

    model = conv_model(in_channels, spatial_size, args)
    print(model)
    model.to(device)

    best_val_loss = np.Inf
    best_corrs = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(model, train_loader, optimizer, device)
        val_loss, corrs, val_df = test(model, val_loader, device)
        if val_loss < best_val_loss:
            print(f"\nSave model at epoch {epoch:03d}, val_loss: {val_loss:.4f}")
            save_weights(model, os.path.join(args.output_dir, f'best_weights.pt'))
            best_val_loss = val_loss
            best_corrs = corrs
        elapsed = (time.time() - start)
        print('Epoch {:03d} finished in : {:.3f} s'.format(epoch, elapsed))
        print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Per-target Spearman R: {:.7f}, Global Spearman R: {:.7f}'.format(
            train_loss, val_loss, corrs['per_target_spearman'], corrs['all_spearman']))

    if test_mode:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_weights.pt')))
        rmse, corrs, test_df = test(model, test_loader, device)
        test_df.to_pickle(os.path.join(args.output_dir, 'test_results.pkl'))
        print('Test RMSE: {:.7f}, Per-target Spearman R: {:.7f}, Global Spearman R: {:.7f}'.format(
            rmse, corrs['per_target_spearman'], corrs['all_spearman']))
        test_file = os.path.join(args.output_dir, f'test_results.txt')
        with open(test_file, 'a+') as out:
            out.write('{}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(
                args.random_seed, rmse, corrs['per_target_spearman'], corrs['all_spearman']))

    return best_val_loss, best_corrs['per_target_spearman'], best_corrs['all_spearman']


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.environ['PSR_DATA'])
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test'])
    parser.add_argument('--output_dir', type=str, default=os.environ['LOG_DIR'])
    parser.add_argument('--unobserved', action='store_true', default=False)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--conv_drop_rate', type=float, default=0.1)
    parser.add_argument('--fc_drop_rate', type=float, default=0.25)
    parser.add_argument('--num_epochs', type=int, default=20)

    parser.add_argument('--num_conv', type=int, default=4)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--no_dropout', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--random_seed', type=int, default=int(np.random.randint(1, 10e6)))

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up output dir
    args.output_dir = os.path.join(args.output_dir, 'psr')
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
