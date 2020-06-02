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

# import atom3d.util.datatypes as dt
import atom3d.util.shard as sh
import examples.cnn3d.feature_resdel as feat


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
            gen = feat.dataset_generator(self.sharded, range(self.num_shards))
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.num_shards / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_shards)
            gen = feat.dataset_generator(self.sharded, range(self.num_shards)[iter_start:iter_end])
        return gen

class cnn_3d_new(nn.Module):

    def __init__(self, nic, noc=20, nf=64, momentum=0.01):
        super(cnn_3d_new, self).__init__()

        # if input channel dim is 1 -- indicates we want to learn embeddings
        self.nic = nic

        self.model= nn.Sequential(
                     # 20  
                    nn.Conv3d(nic, nf, 4, 2, 1, bias=False),
                    nn.BatchNorm3d(nf, momentum=momentum),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.1),

                    # 10 -- consider downsampling earlier in order to speed up training 
                    nn.Conv3d(nf, nf * 2, 3, 1, 1, bias=False),
                    nn.BatchNorm3d(nf * 2, momentum=momentum),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.1),
                    # 10 
                    nn.Conv3d(nf * 2, nf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm3d(nf * 4, momentum=momentum),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.1),

                    # 5
                    nn.Conv3d(nf * 4, nf * 8, 3, 1, 1, bias=False),
                    nn.BatchNorm3d(nf * 8, momentum=momentum),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.1),

                    # 5 
                    nn.Conv3d(nf * 8, nf * 16, 3, 1, 1, bias=False),
                    nn.BatchNorm3d(nf * 16, momentum=momentum),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.1),

                    # 5
                    nn.Conv3d(nf * 16, noc, 5, 1, 0, bias=False),

                    # 1
                )


    def forward(self, input):
        bs = input.size()[0]
        # if input channel dim is 1 -- indicates we want to learn embeddings
        if self.nic == 1: 
            input = self.embed(input)
            input = input.transpose(1, 5)[..., 0]

        output = self.model(input)
        return output.view(bs, -1)


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
    for i, (X,y) in tqdm(enumerate(loader)):
        if i % 1000 == 0:
            print(f'iter {i}, avg acc {np.mean(avg_acc)}')
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        loss = criterion(out, y)
        acc = get_acc(out, y)
        top_k_acc = get_top_k_acc(out, y, k=3)
        losses.append(loss.item())
        avg_acc.append(acc)
        avg_top_k_acc.append(top_k_acc)

    return np.mean(losses), np.mean(avg_acc), np.mean(avg_top_k_acc)




def train(data_dir, device, log_dir, checkpoint=None, seed=None, test_mode=False):
    # logger = logging.getLogger('resdel_log')
    # logging.basicConfig(filename=os.path.join(log_dir, f'train_resdel.log'),level=logging.INFO)

    epochs = 5
    batch_size = 64
    in_channels = 5
    learning_rate = 1e-4
    reg = 5e-6
    shuffle=True
    
    if not os.path.exists(os.path.join(log_dir, 'params.txt')):
        with open(os.path.join(log_dir, 'log.txt'), 'w') as f:
            f.write(f'Epochs: {epochs}\n')
            f.write(f'Batch size: {batch_size}\n')
            f.write(f'Learning rate: {learning_rate}\n')

    train_set = ResDel_Dataset(os.path.join(data_dir, 'train_pdbs@1000/train_pdbs@1000'))
    train_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=4)
    val_set = ResDel_Dataset(os.path.join(data_dir, 'val_pdbs@100/val_pdbs@100'))
    val_loader = data.DataLoader(val_set, batch_size=batch_size, num_workers=4)

    model = cnn_3d_new(nic=in_channels)

    model.to(device)
    if torch.cuda.device_count() > 1:
        print('using', torch.cuda.device_count(), 'GPUs')
        parallel = True
        model = nn.DataParallel(model)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print('loaded pretrained model')


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*torch.cuda.device_count(), weight_decay=reg)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate*torch.cuda.device_count(), weight_decay=reg)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    if test_mode:
        model.eval()
        test_set = ResDel_Dataset(os.path.join(data_dir, 'test_pdbs@100/test_pdbs@100'))
        test_loader = data.DataLoader(val_set, batch_size=batch_size, num_workers=4)
        test_loss, test_acc, test_top_k_acc = test(model, test_loader, criterion, device)
        print('Test loss: {:7f}, Test Accuracy {:.4f}, Top 3 Accuracy {:4f}'.format(test_loss, test_acc, test_top_k_acc))
        return


    best_val_loss = 999
    best_val_idx = 0
    validation_frequency = 10000
    print_frequency = 100

    model.train()

    for epoch in range(1, epochs+1):
        print(f'EPOCH {epoch}\n------------')

        start = time.time()

        for it, (X,y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            if shuffle:
                p = np.random.permutation(batch_size)
                X = X[p]
                y = y[p]

            optimizer.zero_grad()
            out = model(X)
            train_loss = criterion(out, y)
            train_loss.backward()
            optimizer.step()

            # elapsed = time.time() - start
            # print(f'Epoch {epoch}, iter {it}, train loss {train_loss}, avg it/sec {print_frequency / elapsed}')
            # start = time.time()


            if it % print_frequency == 0:
                elapsed = time.time() - start
                print(f'Epoch {epoch}, iter {it}, train loss {train_loss}, avg it/sec {print_frequency / elapsed}')
                start = time.time()
            if (it % validation_frequency == 0) and it > 0: 
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
    parser.add_argument('--data_dir', type=str, default='/oak/stanford/groups/rbaltman/aderry/atom3d/data/residue_deletion')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir

    if log_dir is None:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = os.path.join(args.data_dir, 'cnn3d', 'logs', now)
    else:
        log_dir = os.path.join(args.data_dir, 'cnn3d', 'logs', log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.checkpoint is None:
        checkpoint = os.path.join(args.data_dir, 'CNN_3D_epoch_005_5000_weights.pt')
    if args.mode == 'train':
        train(args.data_dir, device, log_dir, checkpoint)

    if args.mode == 'test':
        train(args.data_dir, device, log_dir, test_mode=True)

