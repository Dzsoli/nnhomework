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
from dataloader import *
from models import *
import pickle

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








n_hidden = 8
# rnn = RNN(n_letters, n_hidden, n_categories)
