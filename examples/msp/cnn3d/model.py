import numpy as np
import torch
import torch.nn as nn


class CNN3D_MSP(nn.Module):
    def __init__(self, in_channels, spatial_size,
                 conv_drop_rate, fc_drop_rate, top_nn_drop_rate,
                 conv_filters, conv_kernel_size,
                 max_pool_positions, max_pool_sizes, max_pool_strides,
                 fc_units,
                 top_fc_units,
                 batch_norm,
                 dropout):
        super(CNN3D_MSP, self).__init__()

        self.base_net, base_features = self.base_network(
            in_channels, spatial_size,
            conv_drop_rate, fc_drop_rate,
            conv_filters, conv_kernel_size,
            max_pool_positions, max_pool_sizes, max_pool_strides,
            fc_units,
            batch_norm,
            dropout
            )

        in_features = 2*base_features
        layers = []
        # Top FCs
        for units in top_fc_units:
            layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU()
                ])
            if batch_norm:
                layers.append(nn.BatchNorm3d(units))
            if dropout:
                layers.append(nn.Dropout(top_nn_drop_rate))
            in_features = units

        # Final FC layer
        layers.append(nn.Linear(in_features, 1))
        self.top_net = nn.Sequential(*layers)

    def base_network(self, in_channels, spatial_size,
                     conv_drop_rate, fc_drop_rate,
                     conv_filters, conv_kernel_size,
                     max_pool_positions, max_pool_sizes, max_pool_strides,
                     fc_units,
                     batch_norm,
                     dropout):
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

        model = nn.Sequential(*layers)
        return model, in_features

    def forward(self, x_original, x_mutated):
        processed_original = self.base_net(x_original)
        processed_mutated = self.base_net(x_mutated)
        x = torch.cat([processed_original, processed_mutated], 1)
        x = self.top_net(x)
        return torch.sigmoid(x).view(-1)
