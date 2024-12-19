import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import scipy.io


class FocusedWaveDataset(Dataset):

    scan_select_seq = {
        '1': [31], # only 1 transmit
        '3': [ 0, 31, 63],
        '5': [ 0, 15, 31, 47, 63],
        '7': [ 0, 10, 21, 31, 42, 52, 63],
        '9': [ 0,  7, 15, 23, 31, 39, 47, 55, 63],
        '11': [ 0,  6, 12, 18, 25, 31, 37, 44, 50, 56, 63]
    }
    n_c = 64
    max_n_scan = 11
    sos_margin = 60
    sos_mean = 1540
    sos_scale = 100

    def __init__(self, data_root, num_scan):
        self.data_root = data_root
        self.file_list = os.listdir(os.path.join(data_root, 'sensor_data'))
        self.num_scan = num_scan

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rf = scipy.io.loadmat(
            os.path.join(self.data_root, 'sensor_data', self.file_list[idx]))
        rf = rf['sensor_data']
        n_scan, n_sensor, n_t = rf.shape

        rf_input = np.zeros((self.max_n_scan, self.n_c, n_t), dtype=np.float32)
        selected_scans = self.scan_select_seq[str(self.num_scan)]
        for i, s in enumerate(selected_scans):
            rf_this = rf[s]
            paste_start = s - n_sensor // 2
            if paste_start < 0:
                crop_start = -paste_start
                paste_start = 0
            else:
                crop_start = 0
            
            paste_end = s + n_sensor - n_sensor // 2
            if paste_end > self.n_c:
                crop_end = n_sensor - (paste_end - self.n_c)
                paste_end = self.n_c
            else:
                crop_end = n_sensor

            rf_input[i, paste_start:paste_end] = rf_this[crop_start:crop_end]

        sos = scipy.io.loadmat(
            os.path.join(self.data_root, 'c0', self.file_list[idx]))
        sos = sos['sos_map'][:, self.sos_margin:-self.sos_margin].T 
        sos = (sos[np.newaxis] - self.sos_mean) / self.sos_scale

        return rf_input, sos


class PlaneWaveDataset(Dataset):

    scan_select_seq = {
        '1': [5], # only 1 transmit
        '3': [ 0, 5, 10],
        '5': [ 0, 2, 5, 8, 10],
        '7': [ 0, 2, 3, 5, 7, 8, 10],
        '9': [ 0,  1, 2, 4, 5, 6, 8, 9, 20],
        '11': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    n_c = 64
    max_n_scan = 11
    sos_margin = 60
    sos_mean = 1540
    sos_scale = 100

    def __init__(self, data_root, num_scan):
        self.data_root = data_root
        self.file_list = os.listdir(os.path.join(data_root, 'plane_wave'))
        self.num_scan = num_scan

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rf = scipy.io.loadmat(
            os.path.join(self.data_root, 'plane_wave', self.file_list[idx]))
        rf = rf['sensor_data_pw']
        n_scan, n_sensor, n_t = rf.shape

        rf_input = np.zeros((self.max_n_scan, self.n_c, n_t), dtype=np.float32)
        selected_scans = self.scan_select_seq[str(self.num_scan)]
        for i, s in enumerate(selected_scans):
            rf_this = rf[s]
            crop_start = (n_sensor - self.n_c) // 2
            crop_end = crop_start + self.n_c
            rf_input[i] = rf_this[crop_start:crop_end]

        sos = scipy.io.loadmat(
            os.path.join(self.data_root, 'c0', self.file_list[idx]))
        sos = sos['sos_map'][:, self.sos_margin:-self.sos_margin].T 
        sos = (sos[np.newaxis] - self.sos_mean) / self.sos_scale

        return rf_input, sos


class ScanLineDataset(Dataset):

    n_c = 64
    max_n_scan = 11
    sos_margin = 60
    sos_mean = 1540
    sos_scale = 100

    def __init__(self, data_root):
        self.data_root = data_root
        self.file_list = os.listdir(os.path.join(data_root, 'scan_line'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rf = scipy.io.loadmat(
            os.path.join(self.data_root, 'scan_line', self.file_list[idx]))
        rf = rf['scan_lines']
        n_scan, n_t = rf.shape

        rf_input = np.zeros((self.max_n_scan, self.n_c, n_t), dtype=np.float32)
        rf_input[0] = rf

        sos = scipy.io.loadmat(
            os.path.join(self.data_root, 'c0', self.file_list[idx]))
        sos = sos['sos_map'][:, self.sos_margin:-self.sos_margin].T 
        sos = (sos[np.newaxis] - self.sos_mean) / self.sos_scale

        return rf_input, sos

if __name__ == '__main__':
    print('---SL---')
    ds = ScanLineDataset('./data/train')
    for i in range(10):
        rf_input, sos = ds[i]
        print('{} {}'.format(rf_input.shape, sos.shape))
    print('---FW---')
    ds = FocusedWaveDataset('./data/train', 3)
    for i in range(10):
        rf_input, sos = ds[i]
        print('{} {}'.format(rf_input.shape, sos.shape))
    print('---PW---')
    ds = PlaneWaveDataset('./data/train', 3)
    for i in range(10):
        rf_input, sos = ds[i]
        print('{} {}'.format(rf_input.shape, sos.shape))
