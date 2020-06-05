import numpy as np
import os
import math
from tqdm import tqdm
import argparse
import datetime
import time
import logging

import dotenv as de
de.load_dotenv()

import sys
sys.path.append('../..')

import torch
import torch.nn as nn
from torch.utils import data
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_add_pool, DataParallel
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F


# import atom3d.util.datatypes as dt
import atom3d.util.shard as sh
import resdel_dataloader as dl


class ResDel_Dataset(data.IterableDataset):
    def __init__(self, sharded):
        self.sharded = sh.load_sharded(sharded)
        self.names = sh.get_keys(sharded)
        self.num_shards = self.sharded.get_num_shards()
        

    def __len__(self):
        return len(self.names)


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            gen = dl.dataset_generator(self.sharded, range(self.num_shards))
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.num_shards / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_shards)
            gen = dl.dataset_generator(self.sharded, range(self.num_shards)[iter_start:iter_end])
        return gen


class Collater(object):
    def __init__(self, follow_batch):
        self.follow_batch = follow_batch

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)

class DataLoader(data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch), **kwargs)


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
        self.fc2 = Linear(hidden_dim, 20)


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
        return self.fc2(x)


def get_acc(logits, label, cm=None):
    pred = torch.argmax(logits, 1)
    acc = float((pred == label).sum(-1)) / label.size()[0]
    return acc

# from pytorch ...
def get_top_k_acc(output, target, k=3):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #res.append(correct_k.mul_(100.0 / batch_size))
        return correct_k.mul_(1.0 / batch_size).item()


@torch.no_grad()
def test(model, loader, criterion, device):
    model.eval()

    losses = []
    avg_acc = []
    avg_top_k_acc = []
    for i, graph in enumerate(loader):
        graph = graph.to(device)
        out = model(graph.x, graph.edge_index, graph.edge_attr.view(-1), graph.batch)
        loss = criterion(out, graph.y)
        acc = get_acc(out, graph.y)
        top_k_acc = get_top_k_acc(out, graph.y, k=3)
        losses.append(loss.item())
        avg_acc.append(acc)
        avg_top_k_acc.append(top_k_acc)

    return np.mean(losses), np.mean(avg_acc), np.mean(avg_top_k_acc)


def train(data_dir, device, log_dir, seed=None, test_mode=False):
    # logger = logging.getLogger('resdel_log')
    # logging.basicConfig(filename=os.path.join(log_dir, f'train_resdel.log'),level=logging.INFO)

    epochs = 5
    batch_size = 64
    in_channels = 5
    learning_rate = 1e-4
    reg = 5e-6
    
    if not os.path.exists(os.path.join(log_dir, 'params.txt')):
        with open(os.path.join(log_dir, 'log.txt'), 'w') as f:
            f.write(f'Epochs: {epochs}\n')
            f.write(f'Batch size: {batch_size}\n')
            f.write(f'Learning rate: {learning_rate}\n')

    train_set = ResDel_Dataset(os.path.join(data_dir, 'train_pdbs@1000/train_pdbs@1000'))
    train_loader = DataLoader(train_set, batch_size=batch_size)#, num_workers=4)
    val_set = ResDel_Dataset(os.path.join(data_dir, 'val_pdbs@100/val_pdbs@100'))
    val_loader = DataLoader(val_set, batch_size=batch_size)#, num_workers=4)

    for graph in train_loader:
        num_features = graph.num_features
        break

    model = GCN(num_features, hidden_dim=64)
    model.to(device)
    # if torch.cuda.device_count() > 1:
    #     print('using', torch.cuda.device_count(), 'GPUs')
    #     parallel = True
    #     model = DataParallel(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*torch.cuda.device_count(), weight_decay=reg)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    best_val_loss = 999
    best_val_idx = 0
    validation_frequency = 10000
    print_frequency = 100

    for epoch in range(1, epochs+1):
        print(f'EPOCH {epoch}\n------------')

        start = time.time()

        for it, graph in enumerate(train_loader):
            graph = graph.to(device)
            optimizer.zero_grad()
            out = model(graph.x, graph.edge_index, graph.edge_attr.view(-1), graph.batch)
            train_loss = criterion(out, graph.y)
            train_loss.backward()
            optimizer.step()


            if it % print_frequency == 0 and it > 0:
                elapsed = time.time() - start
                print(f'Epoch {epoch}, iter {it}, train loss {train_loss}, avg it/sec {print_frequency / elapsed}')
                start = time.time()
            if it % validation_frequency == 0 and it > 0: 
                print('validating...')
                curr_val_loss, val_acc, val_top_k_acc = test(model, val_loader, criterion, device)
                # logger.info('{:03d}\t{}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(epoch, it, train_loss, curr_val_loss, val_acc, val_top_k_acc))
                print('{:03d}\t{}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(epoch, it, train_loss, curr_val_loss, val_acc, val_top_k_acc))

                if curr_val_loss < best_val_loss:

                    # save best validation score and iteration number
                    best_val_loss = curr_val_loss
                    best_val_idx = it
                    # overwrite best model
                    if parallel:
                        torch.save({
                            'epoch': epoch,
                            'iter': it,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_loss,
                            }, os.path.join(log_dir, f'checkpoint_epoch{epoch}_it{it}.pt'))
                    else:
                        torch.save({
                            'epoch': epoch,
                            'iter': it,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_loss,
                            }, os.path.join(log_dir, f'checkpoint_epoch{epoch}_it{it}.pt'))
                    with open(os.path.join(log_dir, 'log.txt'), 'a') as f:
                        f.write('curr best idx \t %s\n' %best_val_idx)

                model.train()

    return best_val_loss


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_dir', type=str, default=O_DIR+'atom3d/data/residue_deletion')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir


    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(args.data_dir, 'gnn', 'logs', now)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        train(args.data_dir, device, log_dir)