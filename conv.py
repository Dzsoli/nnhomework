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

# csv_name = "C:/repos/full_data/i-80.csv"
csv_name = "C:/repos/si_data.csv"
#
my_data = VehicleDataset(csv_name)
my_data.create_objects()

# data1, label = dataloader(my_data, window_size=20, shift=1)
# data, label = dataloader_2(my_data, window_size=30, shift=5)

path = "C:/repos/full_data/"
data = np.reshape(np.load(path + 'dataset.npy'), (-1, 3, 30))[0:1000]
label = np.reshape(np.load(path + 'labels.npy'), (-1, 3))[0:1000]

print("data shape: {}, label shape: {}", data.shape, label.shape)
# print("data2 shape: {}, label2 shape: {}", data2.shape, label2.shape)

# data = data.transpose(0, 1, 2)

step = int(data.shape[0] / 10)

train_data, valid_data, test_data = data[0:8 * step], data[8 * step:9 * step], data[9 * step:data.shape[0]]
train_label, valid_label, test_label = label[0:8 * step], label[8 * step:9 * step], label[9 * step:data.shape[0]]

# print(data[0:15])
# print(label[0:15])

model = CNN()

epochs = 1000
batch_size_train = train_data.shape[0]
batch_size_test = test_data.shape[0]
batch_size_valid = valid_data.shape[0]

loss_fn = nn.BCELoss()
# loss_fn = nn.NLLLoss()
# loss_fn = nn.CrossEntropyLoss()
# parameters = filter(lambda p: p.requires_grad, model.parameters())
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

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

train_loader = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
valid_loader = DataLoader(dataset_valid, batch_size=batch_size_valid, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True)

if use_cuda:
    x_test = x_test.cuda()
    y_test = y_test.cuda()
    model = model.cuda()

los = []
acc = []
val_error = []

for epoch in range(epochs):
    model.train()
    # tic = time.time()
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

    val_acc, val_err = testing2(model, valid_loader, loss_fn)
    acc.append(val_acc)
    val_error.append(val_err)

    model.eval()
    # print('[epoch: {:d}] train_loss: {:.3f}, ({:.1f}s)'.format(epoch, loss.item(), time.time()-tic) )  # pytorch 0.4 and later

plt.plot(los)
plt.ylabel('Training loss')
plt.show()

plt.plot(val_error)
plt.ylabel('Validating loss')
plt.show()

plt.plot(acc)
plt.ylabel('Validating accuracy')
plt.show()

test_acc = testing2(model, test_loader, loss_fn)
print("Testing accuracy is {}", test_acc)
