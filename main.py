from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import autograd
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from dataloader import *
from models import *
import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# si_data = pd.read_csv('../si_data.csv')

csv_name = "../si_data.csv"

dataset = VehicleDataset(csv_name)
dataset.create_objects()
#print(dataset[1].x[1])
#data, labels = dataloader(dataset, 10)

insize = 1
hisize = 10
ousize = 1
lr = 0.005
num_epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model instance
model = LSTM(input_dim=insize, hidden_dim=hisize, output_dim=ousize)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

hist = np.zeros(num_epochs)

print(dataset[1].size)

for t in range(num_epochs):
    for i in range(dataset.__len__()):
        for k in range(dataset[i].size-1):
            print('dataset: {}, size: {}'.format(i, dataset[i].size))
            a = dataset[i].a[k]
            v = dataset[i].v[k]
            dx = dataset[i].x[k+1]-dataset[i].x[k]

            X_train = autograd.Variable(torch.tensor([dx, v, a]))

            if dataset[i].lane_id[k] != dataset[i].lane_id[k+1]:
                y_train = torch.tensor([dataset[i].lane_id[k+1]-dataset[i].lane_id[k]])
            else:
                y_train = torch.tensor([0])

            model.zero_grad()
            model.hidden = model.init_hidden()

            y_pred = model(X_train)
            print(y_pred)
            loss = criterion(y_pred, y_train)
            if t % 100 == 0:
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()

            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()


print('game over')


