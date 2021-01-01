import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """Base 3D convolutional neural network architecture for molecular data.

    :param in_dim: Input dimension.
    :type in_dim: int
    :param out_dim: Output dimension.
    :type out_dim: int
    :param hidden_dim: Base number of hidden units, defaults to 64
    :type hidden_dim: int, optional
    """    
    def __init__(self, in_dim, out_dim, hidden_dim=64, dropout=0.1):
        super(CNN3D, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(in_dim, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv3d(hidden_dim, hidden_dim * 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv3d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv3d(hidden_dim * 4, hidden_dim * 8, 3, 1, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv3d(hidden_dim * 8, hidden_dim * 16, 3, 1, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv3d(hidden_dim * 16, out_dim, 5, 1, 0, bias=False),
        )


    def forward(self, input):
        bs = input.size()[0]

        output = self.model(input)
        return output.view(bs, -1)
