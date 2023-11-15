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
    clnd_train_dt_comb = cylinder_data['clnd_train_dt_comb']
    clnd_train_lb_comb = cylinder_data['clnd_train_lb_comb']
    clnd_train_tg_comb = cylinder_data['clnd_train_tg_comb']
    clnd_test_dt_comb = cylinder_data['clnd_test_dt_comb']
    clnd_test_lb_comb = cylinder_data['clnd_test_lb_comb']
    clnd_test_tg_comb = cylinder_data['clnd_test_tg_comb']
    # print('clnd_test_dt_comb.shape', clnd_test_dt_comb.shape)
    return clnd_train_dt_comb, clnd_train_lb_comb, clnd_train_tg_comb, clnd_test_dt_comb, clnd_test_lb_comb, clnd_test_tg_comb


class vehicle_state_dataset(Dataset):
    def __init__(self, split='train'):
        print('init')
        data_folder = 'E:\Projects\Brake\District\Matlab'

        mat_pth = os.path.join(data_folder, 'light-dark-clinder.mat')
        cylinder_data = scio.loadmat(mat_pth)
        self.clnd_train_dt_comb, self.clnd_train_lb_comb,self.clnd_train_tg_comb, self.clnd_test_dt_comb, self.clnd_test_lb_comb, self.clnd_test_tg_comb\
            = extract_cylinder_mat_data(cylinder_data)


        if split == 'train':
            self.cylinder_data_targ = self.clnd_train_tg_comb
            self.cylinder_data = self.clnd_train_dt_comb
            self.cylinder_array_lab = self.clnd_train_lb_comb

        elif split == 'test':
            self.cylinder_data_targ = self.clnd_test_tg_comb
            self.cylinder_data = self.clnd_test_dt_comb
            self.cylinder_array_lab = self.clnd_test_lb_comb

    def __len__(self):
        return len(self.cylinder_data_targ)

    def __getitem__(self, index):
        cylinder_data = torch.Tensor(self.cylinder_data[index][0])
        cylinder_data_targ = torch.Tensor(self.cylinder_data_targ[index][0])
        cylinder_array_lab = torch.from_numpy(self.cylinder_array_lab[index].astype('int32'))

        return cylinder_data, cylinder_data_targ, cylinder_array_lab

if __name__ == "__main__":
    print('data loader')

    train_dataset = vehicle_state_dataset(split='train')
    test_dataset = vehicle_state_dataset(split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    for i, sample in enumerate(test_dataloader):

        cylinder_data = sample[0]
        cylinder_targ = sample[1]
        cylinder_tot_lab = sample[2]

        print('light data: ', 'cylinder_data.size()', cylinder_data.size(), 'cylinder_targ.size()', cylinder_targ.size(), 'cylinder_tot_lab', cylinder_tot_lab.size())

        print(cylinder_tot_lab)

        print('i', i)

        if i == 1:
            break