"""
Neural network model definitions.
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, in_features: int, negative_slope=0.3):
        super().__init__()

        self.in_features = in_features
        self.negative_slope = negative_slope

        self.flatten = nn.Flatten()
        self.leakyrelu = nn.LeakyReLU(negative_slope=self.negative_slope)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.in_features, 10),
            self.leakyrelu,
            nn.Linear(10, 20),
            self.leakyrelu,
            nn.Linear(20, 10),
            self.leakyrelu,
            nn.Linear(10, 5),
            self.leakyrelu,
            nn.Linear(5, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        xhat = self.linear_relu_stack(x)
        return xhat


class NeuralNetworkClassifier(nn.Module):
    def __init__(self, in_features: int, negative_slope=0.3):
        super().__init__()

        self.in_features = in_features
        self.negative_slope = negative_slope

        self.flatten = nn.Flatten()
        self.leakyrelu = nn.LeakyReLU(negative_slope=self.negative_slope)

        self.sigmoid = nn.Sigmoid()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.in_features, 10),
            self.leakyrelu,
            nn.Linear(10, 20),
            self.leakyrelu,
            nn.Linear(20, 10),
            self.leakyrelu,
            nn.Linear(10, 5),
            self.leakyrelu,
            nn.Linear(5, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        xhat = self.linear_relu_stack(x)
        xhat_sigmoid = self.sigmoid(xhat)
        return xhat_sigmoid


class BiLSTM(nn.Module):
    def __init__(
        self,
        n_features,
        n_hidden,
        seq_len,
        n_layers,
        bidirectional=False,
        dropout=0.70,
        negative_slope=0.3,
    ):
        super(BiLSTM, self).__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.negative_slope = negative_slope

        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1

        self.lstm = nn.LSTM(
            input_size=self.n_features,  # number of features
            hidden_size=self.n_hidden,  # number of features in hidden state
            num_layers=self.n_layers,  # number of LSTM layers
            batch_first=True,  # (N, T, F)
            bidirectional=True,
        )
        self.fc1 = nn.Linear(n_hidden * self.D, self.n_hidden // 2)
        self.fc2 = nn.Linear(self.n_hidden // 2, 1)  # , bias=0 (no bias), bias=1 (bias)
        self.leakyrelu = nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, sequences):

        # LSTM
        _, (hidden, _) = self.lstm(sequences)

        # concatenate the last forward and last backward step
        cat = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)

        # Activation
        rel = self.leakyrelu(cat)

        # FC layer
        dense1 = self.fc1(rel)

        # Activation
        rel1 = self.leakyrelu(dense1)

        # FC layer
        y_pred = self.fc2(rel1)

        return y_pred


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features,
        n_hidden,
        seq_len,
        n_layers,
        bidirectional=False,
        dropout=0.70,
        negative_slope=0.3,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.sigmoid = nn.Sigmoid()

        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1

        self.lstm = nn.LSTM(
            input_size=self.n_features,  # number of features
            hidden_size=self.n_hidden,  # number of features in hidden state
            num_layers=self.n_layers,  # number of LSTM layers
            batch_first=True,  # (N, T, F)
            bidirectional=True,
        )
        self.fc1 = nn.Linear(n_hidden * self.D, self.n_hidden // 2)
        self.fc2 = nn.Linear(self.n_hidden // 2, 1)  # , bias=0 (no bias), bias=1 (bias)
        self.leakyrelu = nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, sequences):

        # LSTM
        _, (hidden, _) = self.lstm(sequences)

        # concatenate the last forward and last backward step
        cat = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)

        # Activation
        rel = self.leakyrelu(cat)

        # FC layer
        dense1 = self.fc1(rel)

        # Activation
        rel1 = self.leakyrelu(dense1)

        # FC layer
        y_pred = self.fc2(rel1)

        # Sigmoid
        y_pred = self.sigmoid(y_pred)

        return y_pred


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        n_features,
        n_hidden,
        seq_len,
        n_layers,
        dropout=0.70,
        bidirectional=True,
        negative_slope=0.3,
    ):
        super(CNNBiLSTM, self).__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.conv_out_layers = 1
        self.negative_slope = negative_slope

        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1

        self.fc1 = nn.Linear(n_hidden * self.D, self.n_hidden // 2)
        self.fc2 = nn.Linear(self.n_hidden // 2, 1)  # bias=0 (no bias), bias=1 (bias)
        self.leakyrelu = nn.LeakyReLU(negative_slope=self.negative_slope)

        self.c1 = nn.Conv1d(
            in_channels=self.n_features,
            out_channels=self.conv_out_layers,
            kernel_size=2,
            stride=1,
        )
        self.lstm = nn.LSTM(
            input_size=self.conv_out_layers,  # number of features
            hidden_size=self.n_hidden,  # number of features in hidden state
            num_layers=self.n_layers,  # number of LSTM layers
            batch_first=True,  # (N, T, F)
            bidirectional=True,
        )

    def forward(self, sequences):

        # Convolution
        conv_out = self.c1(sequences.view(len(sequences), self.n_features, -1))

        # LSTM
        _, (hidden, _) = self.lstm(conv_out.view(len(conv_out), self.seq_len - 1, -1))

        # concatenate the last forward and last backward step
        cat = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)

        # Activation
        rel = self.leakyrelu(cat)

        # FC layer
        dense1 = self.fc1(rel)

        # Activation
        rel1 = self.leakyrelu(dense1)

        # FC layer
        y_pred = self.fc2(rel1)

        return y_pred


class CNNBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features,
        n_hidden,
        seq_len,
        n_layers,
        dropout=0.70,
        bidirectional=True,
        negative_slope=0.3,
    ):
        super(CNNBiLSTM, self).__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.conv_out_layers = 1
        self.negative_slope = negative_slope

        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1

        self.fc1 = nn.Linear(n_hidden * self.D, self.n_hidden // 2)
        self.fc2 = nn.Linear(self.n_hidden // 2, 1)  # bias=0 (no bias), bias=1 (bias)
        self.leakyrelu = nn.LeakyReLU(negative_slope=self.negative_slope)
        self.sigmoid = nn.Sigmoid()

        self.c1 = nn.Conv1d(
            in_channels=self.n_features,
            out_channels=self.conv_out_layers,
            kernel_size=2,
            stride=1,
        )
        self.lstm = nn.LSTM(
            input_size=self.conv_out_layers,  # number of features
            hidden_size=self.n_hidden,  # number of features in hidden state
            num_layers=self.n_layers,  # number of LSTM layers
            batch_first=True,  # (N, T, F)
            bidirectional=True,
        )

    def forward(self, sequences):

        # Convolution
        conv_out = self.c1(sequences.view(len(sequences), self.n_features, -1))

        # LSTM
        _, (hidden, _) = self.lstm(conv_out.view(len(conv_out), self.seq_len - 1, -1))

        # concatenate the last forward and last backward step
        cat = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)

        # Activation
        rel = self.leakyrelu(cat)

        # FC layer
        dense1 = self.fc1(rel)

        # Activation
        rel1 = self.leakyrelu(dense1)

        # FC layer
        dense2 = self.fc2(rel1)

        # Sigmoid
        y_pred = self.sigmoid(dense2)

        return y_pred
