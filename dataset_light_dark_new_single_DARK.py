import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as scio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def extract_cylinder_mat_data(cylinder_data):
    dark_clind_train_dt = cylinder_data['dark_clind_train_dt']
    dark_clind_train_lb = cylinder_data['dark_clind_train_lb']
    dark_clind_train_recon_tg = cylinder_data['dark_clind_train_recon_tg']
    dark_clind_train_tg = cylinder_data['dark_clind_train_tg']
    dark_clind_test_dt = cylinder_data['dark_clind_test_dt']
    dark_clind_test_lb = cylinder_data['dark_clind_test_lb']
    dark_clind_test_recon_tg = cylinder_data['dark_clind_test_recon_tg']
    dark_clind_test_tg = cylinder_data['dark_clind_test_tg']


    return dark_clind_train_dt, dark_clind_train_lb, dark_clind_train_recon_tg, dark_clind_train_tg, \
           dark_clind_test_dt, dark_clind_test_lb, dark_clind_test_recon_tg, dark_clind_test_tg


class vehicle_state_dataset(Dataset):
    def __init__(self, split='train'):
        print('init')
        data_folder = 'path to the .mat file'

        mat_pth = os.path.join(data_folder, 'light-dark-clinder.mat')
        cylinder_data = scio.loadmat(mat_pth)

        self.dark_clind_train_dt, self.dark_clind_train_lb, self.dark_clind_train_recon_tg, self.dark_clind_train_tg, \
        self.dark_clind_test_dt, self.dark_clind_test_lb, self.dark_clind_test_recon_tg, self.dark_clind_test_tg \
            = extract_cylinder_mat_data(cylinder_data)

        if split == 'train':

            self.cylinder_dark_data = self.dark_clind_train_dt
            self.cylinder_dark_targ = self.dark_clind_train_tg
            self.cylinder_dark_lab = self.dark_clind_train_lb
            self.cylinder_dark_recons = self.dark_clind_train_recon_tg

        elif split == 'test':

            self.cylinder_dark_data = self.dark_clind_test_dt
            self.cylinder_dark_targ = self.dark_clind_test_tg
            self.cylinder_dark_lab = self.dark_clind_test_lb
            self.cylinder_dark_recons = self.dark_clind_test_recon_tg

    def __len__(self):
        return len(self.cylinder_dark_targ)

    def __getitem__(self, index):
        #
        dark_cylinder_data = torch.Tensor(self.cylinder_dark_data[index][0])
        dark_cylinder_targ = torch.Tensor(self.cylinder_dark_targ[index][0])
        dark_cylinder_recons = torch.Tensor(self.cylinder_dark_recons[index][0])
        dark_cylinder_array_lab = torch.from_numpy(self.cylinder_dark_lab[index].astype('int32'))

        return dark_cylinder_data, dark_cylinder_targ, dark_cylinder_recons, dark_cylinder_array_lab

if __name__ == "__main__":
    print('data loader')

    train_dataset = vehicle_state_dataset(split='train')
    test_dataset = vehicle_state_dataset(split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    for i, sample in enumerate(test_dataloader):

        dark_cylinder_data = sample[0]
        dark_cylinder_targ = sample[1]
        dark_cylinder_recons = sample[2]
        dark_cylinder_tot_lab = sample[3]

        print('dark data: ', 'cylinder_data.size()', dark_cylinder_data.size(), 'cylinder_targ.size()', dark_cylinder_targ.size(), 'dark_cylinder_recons.size()', dark_cylinder_recons.size(), 'cylinder_tot_lab', dark_cylinder_tot_lab.size())

        print('i', i)

        if i == 1:
            break
