import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """A basic feed-forward neural network (MLP), with tunable hidden layer number and dimension. 
        The number of layers is assumed to be equal to :math:`len(hidden\_dims) + 2`, including the input and output layers.
        Dropout can optionally be specified and is applied after every layer (except output).

        :param in_dim: Input dimension
        :type in_dim: int
        :param hidden_dims: Dimensions of hidden layers (number of hidden units in each layer)
        :type hidden_dims: list
        :param out_dim: Output dimension
        :type out_dim: int
        :param dropout: Dropout probability, defaults to 0.0
        :type dropout: float, optional
        """    
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.0):    
        super(MLP, self).__init__()
        self.dropout = dropout
        self.hidden = nn.ModuleList()
        hidden_dims = [in_dim] + hidden_dims # add in dim to list for simplicity
        for layer in range(len(hidden_dims)-1):
            self.hidden.append(nn.Linear(hidden_dims[layer], hidden_dims[layer+1]))
        
        self.out = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        """Forward method. transforming input feature vector to output dimension.

        :param x: Input feature vector of shape (batch_size, in_dim)
        :type x: torch.FloatTensor
        :return: Output of shape (batch_size, out_dim)
        :rtype: torch.FloatTensor
        """        
        for layer in self.hidden:
            x = F.dropout(F.relu(layer(x)), self.dropout)
        out = self.out(x)
        return out