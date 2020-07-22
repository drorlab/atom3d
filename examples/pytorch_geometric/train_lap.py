#
# GCN torch-geometric training script for psp
# based on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py

import argparse
import datetime
import os
import time

import lap_dataloader as dl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*4)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)
        self.conv4 = GCNConv(hidden_dim*4, hidden_dim*4)
        self.bn4 = nn.BatchNorm1d(hidden_dim*4)
        self.conv5 = GCNConv(hidden_dim*4, hidden_dim*2)
        self.bn5 = nn.BatchNorm1d(hidden_dim*2)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim*2)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = self.bn5(x)
        x = global_mean_pool(x, batch)
        x = F.relu(x)
        # x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        return self.fc1(x)

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim*4, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, 1)

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x).view(-1)


def train(epoch, gcn_model, ff_model, loader, criterion, optimizer, device):
    gcn_model.train()
    ff_model.train()

    start = time.time()

    losses = []
    total = 0
    print_frequency = 10
    for it, (active, inactive) in enumerate(loader):
        active = active.to(device)
        inactive = inactive.to(device)
        optimizer.zero_grad()
        out_active = gcn_model(active.x, active.edge_index, active.edge_attr.view(-1), active.batch)
        out_inactive = gcn_model(inactive.x, inactive.edge_index, inactive.edge_attr.view(-1), inactive.batch)
        output = ff_model(out_active, out_inactive)
        loss = criterion(output, active.y)
        loss.backward()
        # loss_all += loss.item() * active.num_graphs
        # total += active.num_graphs
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

    for it, (active, inactive) in enumerate(loader):
        active = active.to(device)
        inactive = inactive.to(device)
        out_active = gcn_model(active.x, active.edge_index, active.edge_attr.view(-1), active.batch)
        out_inactive = gcn_model(inactive.x, inactive.edge_index, inactive.edge_attr.view(-1), inactive.batch)
        output = ff_model(out_active, out_inactive)
        loss = criterion(output, active.y)
        # loss_all += loss.item() * active.num_graphs
        # total += active.num_graphs
        losses.append(loss.item())
        y_true.extend(active.y.tolist())
        y_pred.extend(output.tolist())
        if it % print_frequency == 0:
            print(f'iter {it}, loss {np.mean(losses)}')

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    return np.mean(losses), auroc, auprc



def train_lap(data_dir, device, log_dir, seed=None, test_mode=False):
    # logger = logging.getLogger('psp_log')

    num_epochs = 20
    batch_size = 16
    hidden_dim = 32
    learning_rate = 1e-5

    

    train_set = dl.LAP_Dataset_PTG(os.path.join(data_dir, 'train'))
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)
    val_set = dl.LAP_Dataset_PTG(os.path.join(data_dir, 'val'))
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=True)

    if not os.path.exists(os.path.join(log_dir, 'params.txt')):
        with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
            f.write(f'Epochs: {num_epochs}\n')
            f.write(f'Batch size: {batch_size}\n')
            f.write(f'Hidden dim: {hidden_dim}\n')
            f.write(f'Learning rate: {learning_rate}')

    for active, inactive in train_loader:
        num_features = active.num_features
        break

    gcn_model = GCN(num_features, hidden_dim=hidden_dim).to(device)
    ff_model = FeedForward(hidden_dim).to(device)

    best_val_loss = 999
    best_val_auroc = 0

    params = [x for x in gcn_model.parameters()] + [x for x in ff_model.parameters()]

    criterion = nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=3,
                                                           min_lr=0.00001)

    print(f'training for {num_epochs} epochs')
    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_loss = train(epoch, gcn_model, ff_model, train_loader, criterion, optimizer, device)
        print('validating...')
        val_loss, auroc, auprc = test(gcn_model, ff_model, val_loader, criterion, device)
        scheduler.step(val_loss)
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
        test_set = dl.LAP_Dataset_PTG(os.path.join(data_dir, 'test'))
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8, shuffle=True)
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
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = '../../data/lap'
    data_dir = SC_DIR+'atom3d/ligand_activity_prediction/graph_pt'
    log_dir = args.log_dir


    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(base_dir, 'logs', now)
        else:
            log_dir = os.path.join(base_dir, 'logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train_lap(data_dir, device, log_dir)
    elif args.mode == 'test':
        test_loss_list = []
        auroc_list = []
        auprc_list = []
        for seed in np.random.randint(0, 1000, size=3):
            print('seed:', seed)
            log_dir = os.path.join(base_dir, 'logs', f'test_{seed}')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            test_loss, auroc, auprc = train_lap(data_dir, device, log_dir, seed, test_mode=True)
            test_loss_list.append(test_loss)
            auroc_list.append(auroc)
            auprc_list.append(auprc)

    print(f'Avg test_loss: {np.mean(test_loss_list)}, St.Dev test_loss {np.std(test_loss_list)}, \
        Avg AUROC {np.mean(auroc_list)}, St.Dev AUROC {np.std(auroc_list)},\
        Avg auprc {np.mean(auprc_list)}, St.Dev auprc {np.std(auprc_list)}')








