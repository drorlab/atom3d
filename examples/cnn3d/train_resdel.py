import argparse
import datetime
import math
import os
import time

import dotenv as de
import numpy as np
from sklearn.metrics import f1_score

de.load_dotenv()

import torch
import torch.nn as nn
from torch.utils import data

# import atom3d.util.datatypes as dt
import atom3d.shard.shard as sh
import examples.cnn3d.feature_resdel as feat



class ResDel_Dataset(data.IterableDataset):
    def __init__(self, sharded, max_shards=None):
        self.sharded = sh.Sharded.load(sharded)
        self.num_shards = self.sharded.get_num_shards()
        if max_shards:
            self.max_shards = max_shards
        else:
            self.max_shards = self.num_shards


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            gen = feat.dataset_generator(self.sharded, range(self.max_shards))

        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.max_shards / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.max_shards)
            gen = feat.dataset_generator(self.sharded, range(self.max_shards)[iter_start:iter_end])
        return gen

class ResDel_Dataset_PT(data.Dataset):
    def __init__(self, path):
        self.path = path


    def __len__(self):
        return len(os.listdir(self.path)) // 2
    
    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.path, f'data_t_{idx}.pt'))
        label = torch.load(os.path.join(self.path, f'label_t_{idx}.pt'))
        return data, label

class cnn_3d_new(nn.Module):

    def __init__(self, nic, noc=20, nf=64):
        super(cnn_3d_new, self).__init__()

        # if input channel dim is 1 -- indicates we want to learn embeddings
        self.nic = nic

        self.model= nn.Sequential(
                     # 20  
                    nn.Conv3d(nic, nf, 4, 2, 1, bias=False),
                    nn.BatchNorm3d(nf),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.1),

                    # 10 -- consider downsampling earlier in order to speed up training 
                    nn.Conv3d(nf, nf * 2, 3, 1, 1, bias=False),
                    nn.BatchNorm3d(nf * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.1),
                    # 10 
                    nn.Conv3d(nf * 2, nf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm3d(nf * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.1),

                    # 5
                    nn.Conv3d(nf * 4, nf * 8, 3, 1, 1, bias=False),
                    nn.BatchNorm3d(nf * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.1),

                    # 5 
                    nn.Conv3d(nf * 8, nf * 16, 3, 1, 1, bias=False),
                    nn.BatchNorm3d(nf * 16),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.1),

                    # 5
                    nn.Conv3d(nf * 16, noc, 5, 1, 0, bias=False),

                    # 1
                )


    def forward(self, input):
        bs = input.size()[0]

        output = self.model(input)
        return output.view(bs, -1)


def get_acc(logits, label, cm=None):
    pred = torch.argmax(logits, 1)
    acc = float((pred == label).sum(-1)) / label.size()[0]
    return acc, pred

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
def test(model, loader, criterion, device, max_steps=None):
    model.eval()

    losses = []
    avg_acc = []
    avg_top_k_acc = []
    y_true = []
    y_pred = []
    for i, (X,y) in enumerate(loader):
        if i % 1000 == 0:
            print(f'iter {i}, avg acc {np.mean(avg_acc)}')
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        loss = criterion(out, y)
        acc, pred = get_acc(out, y)
        top_k_acc = get_top_k_acc(out, y, k=3)
        losses.append(loss.item())
        avg_acc.append(acc)
        avg_top_k_acc.append(top_k_acc)
        y_true.extend(y.tolist())
        y_pred.extend([p.item() for p in pred])
        # if max_steps and i == max_steps:
        #     return np.mean(losses), np.mean(avg_acc), np.mean(avg_top_k_acc), np.mean(f1s)
    
    f1 = f1_score(y_true, y_pred, average='micro')

    return np.mean(losses), np.mean(avg_acc), np.mean(avg_top_k_acc), f1




def train(data_dir, device, log_dir, checkpoint=None, seed=None, test_mode=False):
    # logger = logging.getLogger('resdel_log')
    # logging.basicConfig(filename=os.path.join(log_dir, f'train_resdel.log'),level=logging.INFO)

    epochs = 5
    batch_size = 64
    in_channels = 5
    learning_rate = 1e-4
    reg = 5e-6
    parallel = False

    print('Log dir:', log_dir)
    
    if not os.path.exists(os.path.join(log_dir, 'params.txt')):
        with open(os.path.join(log_dir, 'log.txt'), 'w') as f:
            f.write(f'Epochs: {epochs}\n')
            f.write(f'Batch size: {batch_size}\n')
            f.write(f'Learning rate: {learning_rate}\n')

    train_set = ResDel_Dataset_PT(SC_DIR+'atom3d/residue_deletion/cube_pt/train')
    train_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)
    val_set = ResDel_Dataset_PT(SC_DIR+'atom3d/residue_deletion/cube_pt/val')
    val_loader = data.DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=True)

    model = cnn_3d_new(nic=in_channels)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate*torch.cuda.device_count(), weight_decay=reg)

    model.to(device)


    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    
    if checkpoint:
        cpt = torch.load(checkpoint, map_location=device)
        try:
            model.load_state_dict(cpt['model_state_dict'])
            optimizer.load_state_dict(cpt['optimizer_state_dict'])
            if torch.cuda.device_count() > 1:
                print('using', torch.cuda.device_count(), 'GPUs')
                parallel = True
                model = nn.DataParallel(model)
        except:
            if torch.cuda.device_count() > 1:
                print('using', torch.cuda.device_count(), 'GPUs')
                parallel = True
            model = nn.DataParallel(model)
            model.load_state_dict(cpt)
            # model.load_state_dict(cpt['model_state_dict'])
            # optimizer.load_state_dict(cpt['optimizer_state_dict'])
        print('loaded pretrained model')

    best_val_loss = 999
    best_val_idx = 0
    validation_frequency = 10000
    print_frequency = 1000

    model.train()

    for epoch in range(1, epochs+1):
        print(f'EPOCH {epoch}\n------------')

        start = time.time()

        for it, (X,y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            # if shuffle:
            #     p = np.random.permutation(batch_size)
            #     X = X[p]
            #     y = y[p]

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

        print('validating...')
        curr_val_loss, val_acc, val_top_k_acc, val_f1 = test(model, val_loader, criterion, device, max_steps=1000)
        # logger.info('{:03d}\t{}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(epoch, it, train_loss, curr_val_loss, val_acc, val_top_k_acc))
        print('Epoch {:03d}, iter {}, train loss {:.7f}, val loss {:.7f}, val acc {:.7f}, val top 3 {:.7f}, val F1 {:.3f}\n'.format(epoch, it, train_loss, curr_val_loss, val_acc, val_top_k_acc, val_f1))
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
                    }, os.path.join(log_dir, f'best_weights.pt'))
            else:
                torch.save({
                    'epoch': epoch,
                    'iter': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    }, os.path.join(log_dir, f'best_weights.pt'))
            with open(os.path.join(log_dir, 'log.txt'), 'a') as f:
                f.write('curr best idx \t %s\n' %best_val_idx)

        model.train()
    
    if test_mode:
        print('testing...')
        model = cnn_3d_new(nic=in_channels).to(device)
        model.eval()
        test_set = ResDel_Dataset_PT(SC_DIR+'atom3d/residue_deletion/cube_pt/test_unbalanced')
        test_loader = data.DataLoader(test_set, batch_size=batch_size, num_workers=8)
        # cpt = torch.load(os.path.join(log_dir, f'best_weights.pt'))
        cpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(cpt['model_state_dict'])
        test_loss, test_acc, test_top_k_acc, test_f1 = test(model, test_loader, criterion, device)
        print('Test loss: {:7f}, Test Accuracy {:.4f}, Top 3 Accuracy {:4f}, F1 Score {:4f}'.format(test_loss, test_acc, test_top_k_acc, test_f1))
        return test_loss, test_acc, test_top_k_acc, test_f1

    return best_val_loss


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir

    base_dir = '../../data/residue_deletion'
    data_dir = O_DIR+'atom3d/data/residue_deletion/split'
    if args.checkpoint is None:
        args.checkpoint = os.path.join(data_dir, '../CNN_3D_epoch_004_15000_weights.pt')

    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(base_dir, 'logs_cnn', now)
        else:
            log_dir = os.path.join(base_dir, 'logs_cnn', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train(data_dir, device, log_dir, args.checkpoint)
    elif args.mode == 'test':
        test_loss_list = []
        acc_list = []
        f1_list = []
        for seed in np.random.randint(0, 100, size=3):
            print('seed:', seed)
            log_dir = os.path.join(base_dir, 'logs_cnn', f'test_{seed}')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            test_loss, test_acc, test_top_k_acc, test_f1 = train(data_dir, device, log_dir, args.checkpoint, seed=seed, test_mode=True)
            test_loss_list.append(test_loss)
            acc_list.append(test_acc)
            f1_list.append(test_f1)
        print(f'Avg test_loss: {np.mean(test_loss_list)}, St.Dev test_loss {np.std(test_loss_list)}, \
            Avg accuracy {np.mean(acc_list)}, St.Dev accuracy {np.std(acc_list)},\
            Avg F1 {np.mean(f1_list)}, St.Dev F1 {np.std(f1_list)}')

