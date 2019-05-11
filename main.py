from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from dataloader import *


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# si_data = pd.read_csv('../si_data.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_name = "../si_data.csv"
# window = 4
#
# data_to_load = data_sampler(window, csv_name, data_save=True)
# labels = data_sampler(window, csv_name, data_save=False)
# tt = torch.load('labels.pt')

dataset = VehicleDataset(csv_name)
dataset.create_objects()
print(dataset.vehicle_objects[0])


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 8
# rnn = RNN(n_letters, n_hidden, n_categories)
