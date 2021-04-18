import numpy as np
import torch.nn as nn


class CNN3D_LBA(nn.Module):
    def __init__(self, in_channels, spatial_size,
                 conv_drop_rate, fc_drop_rate,
                 conv_filters, conv_kernel_size,
                 max_pool_positions, max_pool_sizes, max_pool_strides,
                 fc_units,
                 batch_norm=True,
                 dropout=False):
        super(CNN3D_LBA, self).__init__()

        model, out_features = self.base_network(
            in_channels, spatial_size,
            conv_drop_rate, fc_drop_rate,
            conv_filters, conv_kernel_size,
            max_pool_positions, max_pool_sizes, max_pool_strides,
            fc_units,
            batch_norm,
            dropout
            )

        layers = []
        self._add_fc_batch_norm_dropout(out_features, 1, False, dropout,
                                        fc_drop_rate, layers, activation=None)
        self.model = nn.Sequential(
            model,
            nn.Sequential(*layers)
            )

    def base_network(self, in_channels, spatial_size,
                     conv_drop_rate, fc_drop_rate,
                     conv_filters, conv_kernel_size,
                     max_pool_positions, max_pool_sizes, max_pool_strides,
                     fc_units,
                     batch_norm=True,
                     dropout=False):
        layers = []

        if batch_norm:
            layers.append(nn.BatchNorm3d(in_channels))
        # Convs.
        for i in range(len(conv_filters)):
            layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, conv_filters[i], kernel_size=conv_kernel_size, bias=True),
                    nn.ReLU(inplace=True)
                    )
                )
            in_channels = conv_filters[i]
            spatial_size -= (conv_kernel_size - 1)

            if max_pool_positions[i]:
                layers.append(nn.MaxPool3d(max_pool_sizes[i], max_pool_strides[i]))
                spatial_size = int(np.floor((spatial_size - (max_pool_sizes[i]-1) - 1)/max_pool_strides[i] + 1))
            self._add_batch_norm_dropout(in_channels, batch_norm, dropout, conv_drop_rate, layers)

        layers.append(nn.Flatten())
        in_features = in_channels * (spatial_size**3)
        # FC layers.
        for units in fc_units:
            self._add_fc_batch_norm_dropout(in_features, units, batch_norm, dropout,
                                            fc_drop_rate, layers)
            in_features = units

        model = nn.Sequential(*layers)
        return model, in_features


    def _add_batch_norm_dropout(self, in_channels, batch_norm, dropout, drop_rate,
                                layers):
        if batch_norm:
            layers.append(nn.BatchNorm3d(in_channels))
        if dropout:
            layers.append(nn.Dropout(drop_rate))

    def _add_fc_batch_norm_dropout(self, in_features, out_features,
                                   batch_norm, dropout, drop_rate,
                                   layers, activation='relu'):
        if activation == 'relu':
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True)
                    )
                )
        else:
            layers.append(nn.Linear(in_features, out_features))
        self._add_batch_norm_dropout(out_features, batch_norm, dropout, drop_rate, layers)

    def forward(self, x):
        return self.model(x).view(-1)
