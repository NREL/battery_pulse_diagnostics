import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
# from torchvision import transforms


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")



# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, output_size)
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
    
class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BasicRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x , h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out, hn, cn

class DataFrameDataset(Dataset):
    # def __init__(self, h5_path, key, target, transform=None):
    def __init__(self, dataframe, target, key, num_features, transform=None):
        # self.h5_path = h5_path
        # self.key = key
        # self.data_cache = pd.HDFStore(self.h5_path)
        # self.data = self.data_cache[self.key]
        self.data = dataframe
        self.transform = transform
        self.targets = self.data[target]
        self.num_features = num_features
        if key == "Charge_Depleting" or key == "Charge_Sustaining":
            self.depleting_or_sustaining = True
        else:
            self.depleting_or_sustaining = False

        # if self.data['voltage_0.0s'].isna().any():
        #     self.data = self.data.drop(['voltage_0.0s', 'current_0.0s', 'power_0.0s'], axis=1).reset_index(drop=True)
        #     self.targets = self.data[target].reset_index(drop=True)
        # elif self.data['voltage_120.0s'].isna().any():
        #     self.data = self.data.drop(['voltage_120.0s', 'current_120.0s', 'power_120.0s'], axis=1).reset_index(drop=True)
        #     self.targets = self.data[target].reset_index(drop=True)

        self.voltage_min = self.data.filter(regex="voltage").min().min()
        self.voltage_max = self.data.filter(regex="voltage").max().max()
        self.current_min = self.data.filter(regex="current").min().min()
        self.current_max = self.data.filter(regex="current").max().max()
        self.power_min = self.data.filter(regex="power").min().min()
        self.power_max = self.data.filter(regex="power").max().max()

        self.target_min = self.targets.min()
        self.target_max = self.targets.max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        sample = sample.filter(regex="voltage|current|power|control_current|control_voltage|control_power")
        sample = sample.fillna(1e3)
        # if self.depleting_or_sustaining:
        #     num_features = 4
        # else:
        #     num_features = 5
        length = int(len(sample)/self.num_features)
        sample = sample.values.reshape((length, self.num_features))  #reshape to (seq_length, num_features)
        # sample = sample.values.reshape((3, length))  # reshape to (num_features, seq_length)
        sample[:, 0] = (sample[:, 0] - self.voltage_min) / (self.voltage_max - self.voltage_min)
        # sample[:, 1] = (sample[:, 1] - self.current_min) / (self.current_max - self.current_min)
        sample[:, 2] = (sample[:, 2] - self.power_min) / (self.power_max - self.power_min)

        # if self.depleting_or_sustaining: #don't do minmax scaling on control columns?

        # sample[0] = (sample[0] - self.voltage_min) / (self.voltage_max - self.voltage_min)
        # sample[1] = (sample[1] - self.current_min) / (self.current_max - self.current_min)
        # sample[2] = (sample[2] - self.power_min) / (self.power_max - self.power_min)

        target = self.targets.iloc[idx]
        target = (target - self.target_min) / (self.target_max - self.target_min)
        if self.transform:
            sample = self.transform(sample)
        return sample, target.astype("float32")

class ToTensor(object): 


    """Convert pandas dataframe sample to tensor."""
    def __call__(self, sample):
        # return torch.tensor([sample[0], sample[1], sample[2]], dtype=torch.float)
        if sample.shape[1] == 3:
            return torch.transpose(torch.tensor([sample[:, 0], sample[:, 1], sample[:, 2]], dtype=torch.float), 0, 1)
        # return torch.transpose(torch.tensor([sample[:, 0], sample[:, 1], sample[:, 2]], dtype=torch.float), 0, 1)
        if sample.shape[1] == 4:
            return torch.transpose(torch.tensor([sample[:, 0], sample[:, 1], sample[:, 2], sample[:, 3]], dtype=torch.float), 0, 1)
        else:
            return torch.transpose(torch.tensor([sample[:, 0], sample[:, 1], sample[:, 2], sample[:, 3], sample[:, 4]], dtype=torch.float), 0, 1)
    
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    losses =[]
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        #pred is size batch_size x 1, flatten
        pred = pred.flatten()
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return sum(losses)/len(losses)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_mse = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_mse.append(loss_fn(pred.flatten(), y).item())
    avg_test_mse = sum(test_mse) / num_batches
    print(f"Test Error: \n Avg MSE: {avg_test_mse:>8f} \n")
    return avg_test_mse

