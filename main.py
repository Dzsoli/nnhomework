from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils

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

use_cuda = torch.cuda.is_available()
print('use_cuda = {}\n'.format(use_cuda))

seed = 1
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_name = "C:/Users/user/PycharmProjects/si_data.csv"
my_data = VehicleDataset(csv_name)
my_data.create_objects()

data, label = dataloader(my_data, window_size=40, shift=1)
data = data.transpose(0, 1, 2)
train_data, valid_data, test_data = data[0:800], data[800:1100], data[1100:1144]
train_label, valid_label, test_label = label[0:800], label[800:1100], label[1100:1144]

model = CNN()

epochs = 30
batch_size = 12
loss_fn = nn.MSELoss()
# loss_fn = nn.CrossEntropyLoss
# parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = optim.Adam(model.parameters(), lr=0.002)

x_train = torch.from_numpy(train_data).float()
y_train = torch.from_numpy(train_label).float()
x_test = torch.from_numpy(test_data).float()
y_test = torch.from_numpy(test_label).float()
x_valid = torch.from_numpy(valid_data).float()
y_valid = torch.from_numpy(valid_label).float()
# print(x_train[5])

dataset_train = TensorDataset(x_train, y_train)
dataset_valid = TensorDataset(x_valid, y_valid)
dataset_test = TensorDataset(x_test, y_test)
# print(dataset_train[0])

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

if use_cuda:
    x_test = x_test.cuda()
    y_test = y_test.cuda()
    model = model.cuda()

los = []

for epoch in range(epochs):
    model.train()
    tic = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = Variable(inputs), Variable(labels)
        # print(inputs)
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        preds = model(inputs)
        if use_cuda:
            preds = preds.cuda()
        # print(labels.size())
        # print(preds.size())
        # print(preds)
        # print(labels)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        los.append(loss)

    model.eval()
    print('[epoch: {:d}] train_loss: {:.3f}, ({:.1f}s)'.format(epoch, loss.item(), time.time()-tic) )  # pytorch 0.4 and later

plt.plot(los)
plt.ylabel('training')
plt.show()

testing(model, test_loader)

