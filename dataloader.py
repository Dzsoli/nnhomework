import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle

##
# TODO: sávváltások: 5, 5, 7, 12, 13, 21, 31, 32, 41, 41, 44, 45...
##

class VehicleDataset(Dataset):
    """NGSIM vehicle dataset"""

    def __init__(self, csv_file, root_dir=None, transform=None):
        # TODO: megmagyarázni a változó neveket és argumentumokat
        """
        asd
        """
        self.all_data = np.array(pd.read_csv(csv_file, delimiter=',', header=None))
        self.root_dir = root_dir
        self.transform = transform
        self.vehicle_objects = None

    def __len__(self):
        """returns with all the number of frames of the dataset"""
        return len(self.all_data)

    def __getitem__(self, idx):
        """returns the idx_th vehicle"""
        t = self.vehicle_objects[idx]
        return t

    def create_objects(self):
        i = 0
        vehicle_objects = []
        while len(self.all_data) > i:
            total_frames = int(self.all_data[i][2])
            until = i + total_frames
            data = self.all_data[i:until]
            vehicle = VehicleData(data)
            #vehicle.lane_changing()
            #TODO: labeling számítás
            vehicle_objects.append(vehicle)
            i = until
        self.vehicle_objects = vehicle_objects


class VehicleData:
    """class for Vehicles. Instantiate in VehicleDataset.__getitem__"""
    def __init__(self, data):
        # car ID
        self.id = int(data[0, 0])
        # frame ID
        self.frames = data[:, 1]
        # total frame number
        self.size = int(data[0, 2])
        # global time
        self.t = data[:, 3]
        # lateral x coordinate
        self.x = data[:, 4]
        self.x = data_normalize(self.x)
        # Longitudinal y coordinate
        self.y = data[:, 5]
        # Dimensions of the car: Length, Width
        self.dims = data[0, 8:10]
        # Type, 1-motor, 2-car, 3-truck
        self.type = int(data[0, 10])
        # Instantenous velocity
        self.v = data[:, 11]
        self.v = data_normalize(self.v)
        # Instantenous acceleration
        self.a = data[:, 12]
        self.a = data_normalize(self.a)
        # lane ID: 1 is the FARTHEST LEFT. 5 is the FARTHEST RIGHT.
        # 6 is Auxiliary lane for off and on ramp
        # 7 is on ramp
        # 8 is off ramp
        self.lane_id = data[:, 13]
        # [None] if no lane change; [+/-1, frame] if there is a lane change in the specific frame
        # [0, frame_id] or [-1, frame_id] or [1, frame_id]
        self.change_lane = None
        # mean, variance, changes or not?, frame id
        self.labels = None

    def set_change_lane(self, l_change):
        self.change_lane = l_change
"""
    def lane_changing(self):
        l_change = []
        total_frames = self.size

        for i in range(int(total_frames) + 1):
            if (self.lane_id[i + 1] - self.lane_id[i]) != 0:
                l_change.append([self.lane_id[i + 1] - self.lane_id[i],
                                 self.frames[i + 1]])
            else:
                l_change.append([0, self.frames[i + 1]])
        l_change = torch.tensor(np.array(l_change))
        self.set_change_lane(l_change)

    def do_labeling(self):
        labels = []"""


def data_normalize(data):
    avg = np.mean(data)
    std = np.std(data)

    norm_data = (data - avg) / std

    return norm_data


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def dataloader(vehicle_objects, window_size, shift):
    # print("dataset length: {}".format(vehicle_objects.__len__()))
    num_of_parameters = 3
    tensor_idx = 0
    total_size = 572

    lane_change_tensor = np.zeros((total_size, num_of_parameters, window_size))
    lane_keeping_tensor = np.zeros((total_size, num_of_parameters, window_size))
    lab = np.zeros((2 * total_size, 3))
    t = np.zeros((num_of_parameters, window_size))
    tt = np.zeros((num_of_parameters, window_size))

    # print(lane_change_tensor.shape)
    # print(lane_keeping_tensor.shape)
    # print(t.shape)
    # print(tt.shape)

    for vehicle in vehicle_objects:
        # print("Vehicle: {}, size: {}".format(vehicle.id, vehicle.size))
        lane_change_idx, label = lane_change_to_idx(vehicle)

        if (lane_change_idx - 1) > 2 * window_size:
            t[0] = vehicle.x[lane_change_idx - window_size + 1: lane_change_idx + 1]
            t[1] = vehicle.v[lane_change_idx - window_size + 1: lane_change_idx + 1]
            t[2] = vehicle.a[lane_change_idx - window_size + 1: lane_change_idx + 1]

            tt[0] = vehicle.x[lane_change_idx - 2 * window_size + 1: lane_change_idx - 1 * window_size + 1]
            tt[1] = vehicle.v[lane_change_idx - 2 * window_size + 1: lane_change_idx - 1 * window_size + 1]
            tt[2] = vehicle.a[lane_change_idx - 2 * window_size + 1: lane_change_idx - 1 * window_size + 1]

            lane_change_tensor[tensor_idx] = t
            lane_keeping_tensor[tensor_idx] = tt

        if label == -1:
            lab[2 * tensor_idx] = [1, 0, 0]
        elif label == 1:
            lab[2 * tensor_idx] = [0, 0, 1]

        lab[2 * tensor_idx + 1] = [0, 1, 0]

        if lane_change_idx != 0:
            tensor_idx = tensor_idx + 1

    # print(tensor_idx)
    # print(lane_change_tensor[50])
    # print(lane_keeping_tensor[50])

    data_tensor = np.concatenate((lane_keeping_tensor, lane_change_tensor))
    labels_tensor = lab

    # print(data_tensor.shape)
    # print(labels_tensor.shape)

    # print(labels_tensor[0:600])

    return data_tensor, labels_tensor


def lane_change_to_idx(vehicle):
    j = 0
    labels = 0
    lane_change_idx = 0

    while (j < vehicle.size - 1) & (lane_change_idx == 0):
        delta = vehicle.lane_id[j + 1] - vehicle.lane_id[j]
        if delta != 0:
            lane_change_idx = j
            labels = delta
            # print("Lane change idx: {}".format(lane_change_idx))
        j = j + 1

    return lane_change_idx, labels


def testing(model, d_loader):
    print('Testing the network...')
    model.eval()
    model.to('cuda')
    correct = 0
    total = 0

    with torch.no_grad():
        for inp, labs in d_loader:
            inp, labs = inp.to('cuda'), labs.to('cuda')
            outputs = model(inp)
            _, prediction = torch.max(outputs.data, 1)
            total += labs.size(0)

            out_idx = torch.argmax(outputs, 1)
            lab_idx = torch.argmax(labs, 1)

            print(outputs)
            print(labs)
            # print(out_idx)
            # print(lab_idx)
            for k in range(len(out_idx)):
                total = total + 1
                if out_idx[k] == lab_idx[k]:
                    correct = correct + 1

        print('Accuracy on the test set: %d %%' % (100 * correct / total))


