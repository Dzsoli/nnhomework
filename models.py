from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def step(self, inp, hidden=None):
        combined = torch.cat((inp, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def forward(self, inputs, hidden):
        steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, 1))
        for i in range(steps):
            output, hidden = self.step(inputs[i], hidden)
            outputs[i] = output
        return outputs, hidden

    def init_hidden(self):
        # TODO gauss init
        return torch.zeros(1, self.hidden_size)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(self.hidden_size, 3)

    def forward(self, x):
        out, (h_n, h_c) = self.rnn(x, None)
        inp = out[:, -1, :]  # Return output at last time-step
        soft = self.linear(inp)
        out = self.softmax(soft)
        return out


class LSTM2(nn.Module):
    def __init__(self, input_dims, sequence_length, cell_size, output_features=3):
        super(LSTM2, self).__init__()
        self.input_dims = input_dims
        self.sequence_length = sequence_length
        self.cell_size = cell_size
        self.lstm = nn.LSTMCell(input_dims, cell_size)
        self.mlp = nn.Sequential(
            nn.Linear(cell_size, cell_size),
            nn.ReLU(),
            nn.Linear(cell_size, cell_size)
        )
        self.to_output = nn.Linear(cell_size, output_features)

    def forward(self, input):

        h_t, c_t = self.init_hidden(input.size(0))

        outputs = torch.zeroes
        print(input.size())
        print(input[0])
        print(input[1])

        for input_seq in input:
            for frame in input_seq:
                h_t, c_t = self.lstm(frame, (h_t, c_t))
                h_t = self.mlp(h_t)
                # outputs.append(self.to_output(h_t))

        return self.to_output(h_t)  # torch.cat(outputs, dim=1)

    def init_hidden(self, batch_size):
        hidden = Variable(next(self.parameters()).data.new(batch_size, self.cell_size), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(batch_size, self.cell_size), requires_grad=False)
        return hidden.zero_(), cell.zero_()


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size=1, output_dim=3,
                 num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, inputs):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, hidden = self.lstm(inputs.view(len(inputs), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = lstm_out[-1].view(self.batch_size, -1)
        return y_pred.view(-1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            # nn.InstanceNorm1d(16),
            nn.PReLU(16),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=7, stride=1, padding=3),
            # nn.InstanceNorm1d(64),
            nn.PReLU(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=1, padding=4),
            # nn.InstanceNorm1d(128),
            nn.PReLU(128),
            nn.AvgPool1d(kernel_size=3)
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(in_channels=64, out_channels=265, kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        # )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(1280, 560)  # output 3 classes: ...
        self.fc2 = nn.Linear(560, 3)
        self.soft = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.drop_out(x)
        output = self.fc1(x.view(x.size()[0], -1))
        output = self.fc2(output)
        output = self.soft(output)
        return output
