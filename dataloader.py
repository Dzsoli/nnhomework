import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class VehicleDataset(Dataset):
    """NGSIM vehicle dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        # TODO: megmagyarázni a változó neveket és argumentumokat (angolul
        """
        asd
        """
        self.all_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """returns with all the number of frames of the dataset"""
        return len(self.all_data)

    def __getitem__(self, idx):
        """returns the idx_th vehicle"""
        return VehicleData()


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


def data_sampler(window_size, _csv_name, data_save):
    if data_save:
        with open(_csv_name) as f:
            read_csv = csv.reader(f, delimiter=',')
            header = next(read_csv)
            data = []

            for rows in read_csv:
                v_id = rows[0]
                frame = rows[1]
                x = rows[4]
                y = rows[5]
                v = rows[11]
                a = rows[12]
                line = rows[13]
                lane_change = rows[18]

                data.append(
                    [int(v_id), int(frame), float(x), float(y), float(v), float(a), int(line), int(lane_change)])

        data_np = np.array(data)
        data = torch.tensor(data_np)
        DataNames = header

        new_data = torch.zeros((data.size(0), window_size, data.size(1)))
        print("new data size: {}".format(new_data.size()))
        print("original data size: {} x {}".format(data.size(0), data.size(1)))
        # print(type(data))
        # print(data[8000])
        same_flag = True
        i = 0
        j = 0
        while data.size(0) - 10 > i:
            for k in range(window_size):
                if data[i + k, 0] != data[i + k + 1, 0]:
                    same_flag = False
                    break

            if same_flag:
                for t in range(window_size):
                    for z in range(data.size(1)):
                        # new_data[j, t, z].add(data[t+i, z])
                        new_data[j, t, z] = data[t + i, z]
                        # print("{} {}  {}: {}".format(j, t, z, new_data[j, t, z]))
                        # print(data[t+i, z])
                j = j + 1
            else:
                i = i
                # print("autó váltáska")
            # print(new_data[j])
            i = i + 1
            same_flag = True
            # print("{}. ciklus".format(i))
        torch.save(new_data, 'new_data.pt')
        print("Tensor saved!")
        export_data = new_data

        return export_data

    else:
        new_data = torch.load('new_data.pt')
        print("Data has been generated, now i create the labels...")

        _sample = []
        _label = []
        print(new_data.size())
        for k in range(new_data.size(0)):
            sav = 0
            for l in range(new_data.size(1)):
                if new_data[k, l, 7] == 1:  # egyik savvaltas
                    sav = 1
                elif new_data[k, l, 7] == -1:  # masik savvaltas
                    sav = 2
                else:
                    sav = sav
            _label.append(sav)
            _sample.append(k)

        export_data = np.array([_sample, _label])
        torch.save(export_data, 'labels.pt')
        print("Labels has been created!")

        return export_data


csv_name = "../si_data.csv"
window = 4

data_to_load = data_sampler(window, csv_name, data_save=True)
labels = data_sampler(window, csv_name, data_save=False)
tt = torch.load('labels.pt')

print(tt[:, 4000:4500])

