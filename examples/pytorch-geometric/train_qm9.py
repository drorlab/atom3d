import os, sys
import os.path as osp
import argparse
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
import torch_geometric.transforms as T
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
sys.path.append('../..')
from atom3d.mpp.data_qm9_for_ptgeom import GraphQM9




class MyTransform(object):    
    def __init__(self,target):
        self.target = target
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, self.target]
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data



class Net(torch.nn.Module):
    def __init__(self, num_features, dim):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)



def test(model,loader,device,std):
    model.eval()
    error = 0
    for data in loader:
        data = data.to(device)
        error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)



def main(target = 0, dim = 64, prefix='mu'):

    # Set up logging
    try:
        os.mkdir('log')
    except FileExistsError:
        pass
    try:
        os.remove('log/'+prefix+'.log')
    except FileNotFoundError:
        pass
    logging.basicConfig(format='%(message)s', filename='log/'+prefix+'.log', level=logging.DEBUG)

    # Create the data set
    logging.info('Loading the QM9 dataset.\n target: %i, prefix for log files: %s'%(target, prefix))
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/qm9/')
    transform = T.Compose([MyTransform(target), Complete(), T.Distance(norm=False)])
    dataset = GraphQM9(path, transform=transform)
   
    # Normalize targets to mean = 0 and std = 1.
    logging.info('Normalizing the data set.')
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, target].item(), std[:, target].item()
    logging.info(' mean: {:.7f}; standard dev.: {:.7f}'.format(mean, std))

    # Load the indices for the split
    logging.info('Loading split from '+path+'/processed')
    test_indices = np.loadtxt(path+'/processed/processed_test.dat',dtype=int)
    vali_indices = np.loadtxt(path+'/processed/processed_valid.dat',dtype=int)
    train_indices = np.loadtxt(path+'/processed/processed_train.dat',dtype=int)

    test_dataset = dataset[test_indices.tolist()]
    val_dataset = dataset[vali_indices.tolist()]
    train_dataset = dataset[train_indices.tolist()]

    logging.info(' training: %i molecules, validation: %i molecules, test: %i molecules.'%(len(train_indices),len(vali_indices),len(test_indices)))


    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=0.00001)


    # TRAINING
    num_epochs = 300
    logging.info('Starting the training with %i epochs.'%(num_epochs))
    best_val_error = None
    for epoch in range(1,num_epochs+1): 
        lr = scheduler.optimizer.param_groups[0]['lr']

        model.train()

        # Calculate the loss
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            l = F.mse_loss(model(data), data.y)
            l.backward()
            loss_all += l.item() * data.num_graphs
            optimizer.step()
        loss = loss_all / len(train_loader.dataset)
        
        # Calculate the validation error
        val_error = test(model, val_loader, device, std)

        scheduler.step(val_error)

        # Calculate the test error
        if best_val_error is None or val_error <= best_val_error:
            test_error = test(model, test_loader, device, std)
            best_val_error = val_error

        logging.info('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

    logging.info('---------------------------------------------------------------------')
    logging.info('Best validation MAE: {:.7f}, corresp. test MAE: {:.7f}'.format(best_val_error, test_error))

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=64, help='dim')
    parser.add_argument('--target', type=int, default=0, help='target')
    parser.add_argument('--prefix', type=str, default='mu', help='prefix for the log files')
    parser.add_argument('--load', action="store_true", help='load existing model if present (not yet implemented)')
    args = parser.parse_args()

    main(args.target, args.dim, args.prefix)


