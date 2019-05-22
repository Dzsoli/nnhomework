import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # print('lavel:{}, index:{}'.format(label, lane_change_idx))
        if (lane_change_idx - 1) > 2 * window_size:
            batch1 = []
            for k in range(N):

                features[0] = vehicle.x[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]#\
                              #- vehicle.x[lane_change_idx - window_size - k * shift: lane_change_idx - k * shift]

                features[1] = vehicle.v[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]
                features[2] = vehicle.a[lane_change_idx - window_size + 1 - k * shift: lane_change_idx + 1 - k * shift]

                batch1.append(features)

            if label == -1:
                left_seq.append(batch1)
            else:
                right_seq.append(batch1)

        elif lane_change_idx == 0:
            batch2 = []
            for k in range(N):
                features[0] = vehicle.x[lane_change_idx + 1 + k * shift: lane_change_idx + 1 + k * shift + window_size] #- \
                              #vehicle.x[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]
                features[1] = vehicle.v[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]
                features[2] = vehicle.a[lane_change_idx + k * shift: lane_change_idx + k * shift + window_size]

                batch2.append(features)

            keep_seq.append(batch2)

    lab = []
    for i in range(N):
        lab.append(left)
    for i in range(N):
        lab.append(right)
    for i in range(N):
        lab.append(keep)

    for l, r, k in zip(left_seq, right_seq, keep_seq):
        batch3 = np.concatenate((l, r, k), axis=0)
        data.append(batch3)
        label_sequences.append(lab)

    data = np.array(data).transpose((0, 1, 3, 2))
    label_sequences = np.array(label_sequences)
    # print(left_seq[0:3])
    # print(right_seq[0:3])
    # print(keep_seq[0:3])
    return data, label_sequences


def dataloader_20(dataset, window_size, shift):
    vehicle_objects = dataset.vehicle_objects
    number = 0
    number_left = 0
    number_right = 0
    left_iter = []
    right_iter = []
    keep_iter = []
    features = []
    data = []
    N = int(window_size/shift)
    for idx, vehicle in enumerate(vehicle_objects):
        lane_change_idx, labels = lane_change_to_idx(vehicle)
        if lane_change_idx > 3 * window_size:
            # print(vehicle.id)
            if labels == 1:
                number_right += 1
                right_iter.append(idx)
            if labels == -1:
                number_left += 1
                left_iter.append(idx)
        if lane_change_idx == 0:
            keep_iter.append(idx)
            number += 1
    # print('numbers: ', number, number_right, number_left)
    for left, right, keep in zip(left_iter, right_iter, keep_iter):
        # lane change left
        lane_change_idx, labels = lane_change_to_idx(vehicle_objects[left])
        for k in range(N):
            features.clear()
            index = lane_change_idx - 2 * window_size + k * shift
            features.append(vehicle_objects[left].x[index: index + window_size]
                            - vehicle_objects[left].x[index - 1: index + window_size - 1])
            features.append(vehicle_objects[left].v[index: index + window_size])
            features.append(vehicle_objects[left].a[index: index + window_size])
            data.append(features)

        # lane change right
        lane_change_idx, labels = lane_change_to_idx(vehicle_objects[right])
        for k in range(N):
            features.clear()
            index = lane_change_idx - 2 * window_size + k * shift
            features.append(vehicle_objects[right].x[index: index + window_size]
                            - vehicle_objects[right].x[index - 1: index + window_size - 1])
            features.append(vehicle_objects[right].v[index: index + window_size])
            features.append(vehicle_objects[right].a[index: index + window_size])
            data.append(features)

        # lane keeping
        _, labels = lane_change_to_idx(vehicle_objects[keep])
        first_idx = 3 * window_size
        for k in range(N):
            features.clear()
            index = first_idx - 2 * window_size + k * shift
            features.append(vehicle_objects[keep].x[index: index + window_size]
                            - vehicle_objects[keep].x[index - 1: index + window_size - 1])
            features.append(vehicle_objects[keep].v[index: index + window_size])
            features.append(vehicle_objects[keep].a[index: index + window_size])
            data.append(features)
    data = np.array(data).transpose((0, 2, 1))
    print(data.shape)
    # label creation
    leftlab = [1,0,0]
    keeplab = [0,1,0]
    rightlab = [0,0,1]
    lab = []
    for i in range(number_right):
        for j in range(N):
            lab.append(leftlab)
        for j in range(N):
            lab.append(rightlab)
        for j in range(N):
            lab.append(keeplab)

    label = np.array(lab)
    # print('shape: ', data.shape, label.shape)
    print(data[0:4])
    return data, label

def dataloader_2(dataset, window_size, shift):
    vehicle_objects = dataset.vehicle_objects
    number = 0
    number_left = 0
    number_right = 0
    left_iter = []
    right_iter = []
    keep_iter = []
    total_idx = 0
    features = np.zeros((3, window_size))
    data = np.zeros((1350, 3, window_size))
    N = int(window_size/shift)
    for idx, vehicle in enumerate(vehicle_objects):
        lane_change_idx, labels = lane_change_to_idx(vehicle)
        if lane_change_idx > 3 * window_size:
            # print(vehicle.id)
            if labels == 1:
                number_right += 1
                right_iter.append(idx)
            if labels == -1:
                number_left += 1
                left_iter.append(idx)
        if lane_change_idx == 0:
            keep_iter.append(idx)
            number += 1
    # print('numbers: ', number, number_right, number_left)
    for left, right, keep in zip(left_iter, right_iter, keep_iter):
        # lane change left
        lane_change_idx, labels = lane_change_to_idx(vehicle_objects[left])
        for k in range(N):
            features[0] = 0
            index = lane_change_idx - 2 * window_size + k * shift + 1
            features[0] = (vehicle_objects[left].x[index: index + window_size]
                           - vehicle_objects[left].x[index - 1: index + window_size - 1])
            features[1] = (vehicle_objects[left].v[index: index + window_size])
            features[2] = (vehicle_objects[left].a[index: index + window_size])
            # print(features)
            data[total_idx] = features
            total_idx += 1
        # print("K")
        # lane change right
        lane_change_idx, labels = lane_change_to_idx(vehicle_objects[right])
        for k in range(N):
            features[0] = 0
            index = lane_change_idx - 2 * window_size + k * shift + 1
            features[0] = (vehicle_objects[right].x[index: index + window_size]
                            - vehicle_objects[right].x[index - 1: index + window_size - 1])
            features[1] = (vehicle_objects[right].v[index: index + window_size])
            features[2] = (vehicle_objects[right].a[index: index + window_size])
            data[total_idx] = features
            total_idx += 1
        # lane keeping
        _, labels = lane_change_to_idx(vehicle_objects[keep])
        first_idx = 3 * window_size
        for k in range(N):
            features[0] = 0
            index = first_idx - 2 * window_size + k * shift + 1
            features[0] = (vehicle_objects[keep].x[index: index + window_size]
                            - vehicle_objects[keep].x[index - 1: index + window_size - 1])
            features[1] = (vehicle_objects[keep].v[index: index + window_size])
            features[2] = (vehicle_objects[keep].a[index: index + window_size])
            data[total_idx] = features
            total_idx += 1

    # print("data shape", data.shape)
    # print(data[0:20])
    # data = np.array(data)
    data = data.transpose((0, 2, 1))
    # label creation
    leftlab = [1, 0, 0]
    keeplab = [0, 1, 0]
    rightlab = [0, 0, 1]
    lab = []
    for i in range(number_right):
        for j in range(N):
            lab.append(leftlab)
        for j in range(N):
            lab.append(rightlab)
        for j in range(N):
            lab.append(keeplab)

    label = np.array(lab)
    # print('shape: ', data.shape, label.shape)ű
    return data, label


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

def testing(model, data, labels):
    print('Testing the network...')
    model.eval()
    model.to(device)
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