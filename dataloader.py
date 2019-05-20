import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle

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
        return len(self.vehicle_objects)

    def __getitem__(self, idx):
        """returns the idx_th vehicle"""
        return self.vehicle_objects[idx]

    def create_objects(self):
        i = 0
        vehicle_objects = []
        while len(self.all_data) > i:
            total_frames = int(self.all_data[i][2])
            until = i + total_frames
            data = self.all_data[i:until]
            vehicle = VehicleData(data)
            # vehicle.lane_changing()
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
        # Longitudinal y coordinate
        self.y = data[:, 5]
        # Dimensions of the car: Length, Width
        self.dims = data[0, 8:10]
        # Type, 1-motor, 2-car, 3-truck
        self.type = int(data[0, 10])
        # Instantenous velocity
        self.v = data[:, 11]
        # Instantenous acceleration
        self.a = data[:, 12]
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

    def lane_changing(self):
        l_change = []
        total_frames = self.size

        for i in range(int(total_frames) - 1):
            if (self.lane_id[i + 1] - self.lane_id[i]) != 0:
                l_change.append([self.lane_id[i + 1] - self.lane_id[i],
                                 self.frames[i + 1]])
            else:
                l_change.append([0, self.frames[i + 1]])
        l_change = np.array(l_change)
        self.set_change_lane(l_change)

    def do_labeling(self):
        labels = []


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def dataloader(vehicle_objects, window_size, shift):
    # print("dataset length: {}".format(vehicle_objects.__len__()))
    num_of_parameters = 3
    tensor_idx = 0
    total_size = 572
    N = int(window_size / shift)

    lane_change_tensor = np.zeros((total_size, num_of_parameters, window_size))
    lane_keeping_tensor = np.zeros((total_size, num_of_parameters, window_size))

    features = np.zeros((num_of_parameters, window_size))
    tt = np.zeros((num_of_parameters, window_size))

    left_seq = []
    right_seq = []
    keep_seq = []
    label_sequences = []
    data = []
    left = [1., 0., 0.]
    right = [0., 0., 1.]
    keep = [0., 1., 0.]
    # print(lane_change_tensor.shape)
    # print(lane_keeping_tensor.shape)
    # print(features.shape)
    # print(tt.shape)

    for vehicle in vehicle_objects:
        # print("Vehicle: {}, size: {}".format(vehicle.id, vehicle.size))
        lane_change_idx, label = lane_change_to_idx(vehicle)

        if (lane_change_idx - 1) > 2 * window_size:
            batch = []
            for k in range(N):

                features[0] = vehicle.x[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]\
                              - vehicle.x[lane_change_idx - window_size - k * shift: lane_change_idx - k * shift]

                features[1] = vehicle.v[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]
                features[2] = vehicle.a[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]

                batch.append(features)

            if label == -1:
                left_seq.append(batch)
            else:
                right_seq.append(batch)

        elif lane_change_idx == 0:
            batch = []
            for k in range(N):
                features[0] = vehicle.x[lane_change_idx + 1 + k * shift: lane_change_idx + 1 + k * shift + window_size] - \
                              vehicle.x[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]
                features[1] = vehicle.v[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]
                features[2] = vehicle.a[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]

                batch.append(features)

            keep_seq.append(batch)

    lab = []
    for i in range(N):
        lab.append(left)
    for i in range(N):
        lab.append(right)
    for i in range(N):
        lab.append(keep)

    for l, r, k in zip(left_seq, right_seq, keep_seq):
        batch = np.concatenate((l, r, k), axis=0)
        data.append(batch)
        label_sequences.append(lab)

    data = np.array(data).transpose((0, 1, 3, 2))
    label_sequences = np.array(label_sequences)
    return data, label_sequences


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
