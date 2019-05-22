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
csv_name = "../si_data.csv"

lr = 0.005
num_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = VehicleDataset(csv_name)
dataset.create_objects()
data, labels = dataloader_2(dataset, 30, 5)

data = torch.from_numpy(data).float().to(device)
labels = torch.LongTensor(labels).to(device=device, dtype=torch.float)

model = SimpleLSTM(3, 10)
model = model.to(device)
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(3000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    err = loss(out, labels)
    if epoch % 50 == 0:
        print('out: {}'.format(out[10]), 'label: {}'.format(labels[10]), 'product: {}'.format(out[10]*labels[10]))
    err.backward()
    optimizer.step()

    # print('epoch: {}, loss: {}'.format(epoch, err))
