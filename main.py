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



# insize = 1
# hisize = 10
# ousize = 1
lr = 0.005
num_epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data, labels = dataloader(dataset.vehicle_objects, 30, 5)
data = torch.from_numpy(data).float()
print('data', data.shape)
labels = torch.LongTensor(labels)
print('labels',labels.shape)
model = SimpleLSTM(3, 10)
model = model.to(device)
model2 = LSTM2(3, 30, 5, 3)
model2 = model2.to(device)
# loss = nn.CrossEntropyLoss()
# loss = nn.MultiLabelSoftMarginLoss()
# loss = nn.NLLLoss()
loss = nn.MSELoss()
optimizer = optim.Adam(model2.parameters(), lr=lr)

for epoch in range(num_epochs):
    model2.train()
    for i in range(87):
        print(i)
        input_batch = data[i].to(device)
        input_label = labels[i].to(device=device, dtype=torch.float)
        print(input_batch.size())
        # print(torch.chunk(input_batch, 30, dim=2)[0].squeeze(2))
        out = model2(input_batch)
        # print(out)
        # print(input_label)
        err = loss(out, input_label)
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

        print('epoch: {}, loss: {}'.format(epoch, err))
print(input_batch[5])
print(out[5])
print(input_label[5])

print(input_batch[6])
print(out[6])
print(input_label[6])

print(input_batch[7])
print(out[7])
print(input_label[7])


# print(len(data))
# print(len(labels))
# print(data[0])
# print(labels[0])
# print(data.shape)


