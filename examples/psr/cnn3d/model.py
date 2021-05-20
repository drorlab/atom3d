import numpy as np
import torch.nn as nn


class CNN3D_PSR(nn.Module):
    def __init__(self, in_channels, spatial_size,
                 conv_drop_rate, fc_drop_rate,
                 conv_filters, conv_kernel_size,
                 max_pool_positions, max_pool_sizes, max_pool_strides,
                 fc_units,
                 batch_norm=True,
                 dropout=False):
        super(CNN3D_PSR, self).__init__()

        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm3d(in_channels))

        # Convs
        for i in range(len(conv_filters)):
            layers.extend([
                nn.Conv3d(in_channels, conv_filters[i],
                          kernel_size=conv_kernel_size,
                          bias=True),
                nn.ReLU()
                ])
            spatial_size -= (conv_kernel_size - 1)
            if max_pool_positions[i]:
                layers.append(nn.MaxPool3d(max_pool_sizes[i], max_pool_strides[i]))
                spatial_size = int(np.floor((spatial_size - (max_pool_sizes[i]-1) - 1)/max_pool_strides[i] + 1))
            if batch_norm:
                layers.append(nn.BatchNorm3d(conv_filters[i]))
            if dropout:
                layers.append(nn.Dropout(conv_drop_rate))
            in_channels = conv_filters[i]

        layers.append(nn.Flatten())
        in_features = in_channels * (spatial_size**3)
        # FC layers
        for units in fc_units:
            layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU()
                ])
            if batch_norm:
                layers.append(nn.BatchNorm3d(units))
            if dropout:
                layers.append(nn.Dropout(fc_drop_rate))
            in_features = units

        # Final FC layer
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).view(-1)
