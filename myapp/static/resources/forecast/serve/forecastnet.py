""" for prediction, just concat every day data (weekly) sorted by state, date and
turn it into a single row. Once you get a weekly (6 day) row, append to the archive train data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ForecastNet(nn.Module):
    def __init__(self, inp_dim, out_dim, hid_dims):
        super(ForecastNet, self).__init__()

        self.out_dim = out_dim
        self.fc1 = nn.Linear(inp_dim, hid_dims[0])
        self.fc2 = nn.Linear(hid_dims[0], hid_dims[1])
        self.bn = nn.BatchNorm1d(hid_dims[1])
        self.fc_out = nn.Linear(hid_dims[1], out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn(x)
        x = F.relu(self.fc_out(x))

        return x
