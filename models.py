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
import random
import time
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 30


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
        #TODO gauss init
        return torch.zeros(1, self.hidden_size)


class LSTM(nn.Module):
    #TODO: be kell fejezni
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.Linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def step(self, inp, hidden=None):

        return output, hidden

    def forward(self, inputs, hidden):
        steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, 1))
        outputs, hidden = self.lstm(inputs.view(len(inputs), -1))
        return outputs, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        output, hidden = self.gru(inputs, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        output = F.relu(inputs)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


"""Helper functions"""


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

"""
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
"""


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # size after pooling?
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.fc1 = nn.Linear(192, 80)  # output 3 classes: ...
        self.fc2 = nn.Linear(80, 3)
        self.soft = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.fc1(x.view(x.size()[0], -1))
        output = self.fc2(output)
        output = self.soft(output)
        return output


class DilatedNet(nn.Module):
    def __init__(self, num_securities=5, hidden_size=64, dilation=2, t=10):
        """:param num_securities: int, number of stocks
        :param hidden_size: int, size of hidden layers
        :param dilation: int, dilation value
        :param T: int, number of look back points"""

        super(DilatedNet, self).__init__()
        self.dilation = dilation
        self.hidden_size = hidden_size
        # First Layer
        self.dilated_conv1 = nn.Conv1d(num_securities, hidden_size, kernel_size=2, dilation=self.dilation)
        self.relu1 = nn.ReLU()
        # Layer 2
        self.dilated_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu2 = nn.ReLU()
        # Layer 3
        self.dilated_conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu3 = nn.ReLU()
        # Layer 4
        self.dilated_conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu4 = nn.ReLU()
        # Output layer
        self.conv_final = nn.Conv1d(hidden_size, num_securities, kernel_size=1)
        self.t = t

    def forward(self, x):
        """ :param x: Pytorch Variable, batch_size x n_stocks x T
            :return:"""
        # First layer
        out = self.dilated_conv1(x)
        out = self.relu1(out)
        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)
        # Layer 3:
        out = self.dilated_conv3(out)
        out = self.relu3(out)
        # Layer 4:
        out = self.dilated_conv4(out)
        out = self.relu4(out)
        # Final layer
        out = self.conv_final(out)
        out = out[:, :, -1]
        return out


class DilatedNet2D(nn.Module):
    def __init__(self, hidden_size=64, dilation=1, t=10):
        """:param hidden_size: int, size of hidden layers
        :param dilation: int, dilation value
        :param T: int, number of look back points"""
        super(DilatedNet2D, self).__init__()
        self.dilation = dilation
        self.hidden_size = hidden_size
        # First Layer
        self.dilated_conv1 = nn.Conv2d(1, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu1 = nn.ReLU()
        # Layer 2
        self.dilated_conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu2 = nn.ReLU()
        # Layer 3
        self.dilated_conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu3 = nn.ReLU()
        # Layer 4
        self.dilated_conv4 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu4 = nn.ReLU()
        # Output layer
        self.conv_final = nn.Conv2d(hidden_size, 1, kernel_size=(1, 2))
        self.t = t

    def forward(self, x):
        """:param x: Pytorch Variable, batch_size x 1 x n_stocks x T
        :return:"""
        # First layer
        out = self.dilated_conv1(x)
        out = self.relu1(out)
        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)
        # Layer 3:
        out = self.dilated_conv3(out)
        out = self.relu3(out)
        # Layer 4:
        out = self.dilated_conv4(out)
        out = self.relu4(out)
        # Final layer
        out = self.conv_final(out)
        out = out[:, :, :, -1]

        return out