"""
Generate loss wrapper for multi-task learning
"""
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------classification-----------------------

def cylinder_loss(sig_data, lab_array):
    # print('sig_data.size', len(sig_data),'sig_data[1].size:',sig_data[1].size(), 'lab_array.size', lab_array.size())

    criterion_class = nn.CrossEntropyLoss()
    cylinder_loss = torch.Tensor([0]).to(device)
    for i in range(4):
        # print(i)
        sig = sig_data[i]
        # print('in loop i:', i, 'sig.size():', sig.size(), 'lab_array[:,i].size: ', lab_array[:,i].size())
        # print('lab_array[:,i]', i, lab_array[:,i])
        cylinder_loss += criterion_class(sig, lab_array[:,i])

    # print('cylinder_loss: ', cylinder_loss)
    return cylinder_loss


def can_loss(sig_data, lab_array):
    criterion_class = nn.CrossEntropyLoss()
    can_loss = torch.Tensor([0]).to(device)
    for i in range(4):
        # print(i)
        can_loss += criterion_class(sig_data[i], lab_array[:,i])
    return can_loss


def motor_loss(sig_data, lab_array):
    criterion_class = nn.CrossEntropyLoss()
    motor_loss = torch.Tensor([0]).to(device)
    for i in range(3):
        # print(i)
        motor_loss += criterion_class(sig_data[i], lab_array[:,i])
    return motor_loss


def battery_loss(sig_data, lab_array):
    criterion_class = nn.CrossEntropyLoss()
    battery_loss = torch.Tensor([0]).to(device)
    for i in range(3):
        # print(i)
        battery_loss += criterion_class(sig_data[i], lab_array[:,i])
    return battery_loss


def Sub_Sys_loss(cylinder_sub_sys_output, can_sub_sys_output, motor_sub_sys_output, battery_sub_sys_output,
                 cylinder_tot_lab, can_tot_lab, motor_tot_lab, battery_tot_lab):

    criterion_class = nn.BCEWithLogitsLoss()

    sub_sys_loss1 = criterion_class(cylinder_sub_sys_output, cylinder_tot_lab)
    sub_sys_loss2 = criterion_class(can_sub_sys_output, can_tot_lab)
    sub_sys_loss3 = criterion_class(motor_sub_sys_output, motor_tot_lab)
    sub_sys_loss4 = criterion_class(battery_sub_sys_output, battery_tot_lab)

    sub_sys_loss = sub_sys_loss1 + sub_sys_loss2 + sub_sys_loss3 + sub_sys_loss4

    return sub_sys_loss

# -----------------regression-----------------------

def cylinder_loss_reg(sig_data, lab_array):
    criterion_reg = nn.MSELoss()
    cylinder_loss = torch.Tensor([0]).to(device)
    for i in range(4):
        # print(i)
        # cylinder_loss += criterion_reg(sig_data[:,:,i], lab_array[:,:,i])
        cylinder_loss += criterion_reg(sig_data[i], lab_array[:,:,i])

    return cylinder_loss


def can_loss_reg(sig_data, lab_array):
    criterion_reg = nn.MSELoss()
    can_loss = torch.Tensor([0]).to(device)
    for i in range(4):
        # print(i)
        # can_loss += criterion_reg(sig_data[:,:,i], lab_array[:,:,i])
        can_loss += criterion_reg(sig_data[i], lab_array[:, :, i])
    return can_loss


def motor_loss_reg(sig_data, lab_array):
    criterion_reg = nn.MSELoss()
    motor_loss = torch.Tensor([0]).to(device)
    for i in range(3):
        # print(i)
        # motor_loss += criterion_reg(sig_data[:,:,i], lab_array[:,:,i])
        motor_loss += criterion_reg(sig_data[i], lab_array[:, :, i])
    return motor_loss


def battery_loss_reg(sig_data, lab_array):
    criterion_reg = nn.MSELoss()
    battery_loss = torch.Tensor([0]).to(device)
    for i in range(3):
        # print(i)
        # battery_loss += criterion_reg(sig_data[:,:,i], lab_array[:,:,i])
        battery_loss += criterion_reg(sig_data[i], lab_array[:,:,i])

    return battery_loss


# -----------------MultiTaskLossWrapper-----------------------
class MultiTaskLossWrapper_2(nn.Module):
    def __init__(self, task_num=2):
        super(MultiTaskLossWrapper_2, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, cylinder_reg_loss, cylinder_cls_loss):

        precision0 = torch.exp(-self.log_vars[0])
        loss0_p = precision0 * cylinder_reg_loss + self.log_vars[0]

        # print('loss1', loss1)
        precision1 = torch.exp(-self.log_vars[1])
        loss1_p = precision1 * cylinder_cls_loss + self.log_vars[1]

        tot_loss = loss0_p + loss1_p

        return tot_loss, self.log_vars


class MultiTaskLossWrapper_3(nn.Module):
    def __init__(self, task_num=3):
        super(MultiTaskLossWrapper_3, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, cylinder_reg_loss, cylinder_recons_loss, cylinder_cls_loss):

        precision0 = torch.exp(-self.log_vars[0])
        loss0_p = precision0 * cylinder_reg_loss + self.log_vars[0]

        # print('loss1', loss1)
        precision1 = torch.exp(-self.log_vars[1])
        loss1_p = precision1 * cylinder_cls_loss + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2_p = precision2 * cylinder_recons_loss + self.log_vars[2]

        tot_loss = loss0_p + loss1_p + loss2_p

        return tot_loss, self.log_vars


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num=6):
        super(MultiTaskLossWrapper, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, cylinder_loss, can_loss, motor_loss, battery_loss, binary_loss, sub_sys_loss):

        precision0 = torch.exp(-self.log_vars[0])
        loss0_p = precision0 * cylinder_loss + self.log_vars[0]

        # print('loss1', loss1)
        precision1 = torch.exp(-self.log_vars[1])
        loss1_p = precision1 * can_loss + self.log_vars[1]
        #
        precision2 = torch.exp(-self.log_vars[2])
        loss2_p = precision2 * motor_loss + self.log_vars[2]

        precision3 = torch.exp(-self.log_vars[3])
        loss3_p = precision3 * battery_loss + self.log_vars[3]

        precision4 = torch.exp(-self.log_vars[4])
        loss4_p = precision4 * binary_loss + self.log_vars[4]

        precision5 = torch.exp(-self.log_vars[5])
        loss5_p = precision5 * sub_sys_loss + self.log_vars[5]

        tot_loss = loss0_p + loss1_p + loss2_p + loss3_p + loss4_p + loss5_p

        return tot_loss, self.log_vars

#
# class MultiTaskLossWrapper(nn.Module):
#     def __init__(self, task_num=9):
#         super(MultiTaskLossWrapper, self).__init__()
#         self.log_vars = nn.Parameter(torch.zeros(task_num))
#
#     def forward(self, cylinder_loss, can_loss, motor_loss, battery_loss, binary_loss,
#                 cylinder_loss_reg, can_loss_reg, motor_loss_reg, battery_loss_reg):
#
#         precision0 = torch.exp(-self.log_vars[0])
#         loss0_p = precision0 * cylinder_loss + self.log_vars[0]
#
#         # print('loss1', loss1)
#         precision1 = torch.exp(-self.log_vars[1])
#         loss1_p = precision1 * can_loss + self.log_vars[1]
#         #
#         precision2 = torch.exp(-self.log_vars[2])
#         loss2_p = precision2 * motor_loss + self.log_vars[2]
#
#         precision3 = torch.exp(-self.log_vars[3])
#         loss3_p = precision3 * battery_loss + self.log_vars[3]
#
#         precision4 = torch.exp(-self.log_vars[4])
#         loss4_p = precision4 * binary_loss + self.log_vars[4]
#
#         precision5 = torch.exp(-self.log_vars[5])
#         loss5_p = precision5 * cylinder_loss_reg + self.log_vars[5]
#
#         # print('loss1', loss1)
#         precision6 = torch.exp(-self.log_vars[6])
#         loss6_p = precision6 * can_loss_reg + self.log_vars[6]
#         #
#         precision7 = torch.exp(-self.log_vars[7])
#         loss7_p = precision7 * motor_loss_reg + self.log_vars[7]
#
#         precision8 = torch.exp(-self.log_vars[8])
#         loss8_p = precision8 * battery_loss_reg + self.log_vars[8]
#
#         tot_loss = loss0_p + loss1_p + loss2_p + loss3_p + loss4_p + loss5_p + loss6_p + loss7_p + loss8_p
#
#         return tot_loss


class MultiTaskLossWrapper_reg(nn.Module):
    def __init__(self, task_num=4):
        super(MultiTaskLossWrapper_reg, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, cylinder_loss_reg, can_loss_reg, motor_loss_reg, battery_loss_reg):

        precision5 = torch.exp(-self.log_vars[0])
        loss5_p = precision5 * cylinder_loss_reg + self.log_vars[0]

        # print('loss1', loss1)
        precision6 = torch.exp(-self.log_vars[1])
        loss6_p = precision6 * can_loss_reg + self.log_vars[1]
        #
        precision7 = torch.exp(-self.log_vars[2])
        loss7_p = precision7 * motor_loss_reg + self.log_vars[2]

        precision8 = torch.exp(-self.log_vars[3])
        loss8_p = precision8 * battery_loss_reg + self.log_vars[3]

        # tot_loss = loss5_p + loss6_p + loss7_p + loss8_p

        tot_loss = cylinder_loss_reg + can_loss_reg + motor_loss_reg + battery_loss_reg

        return tot_loss


if __name__ == "__main__":
    inputs = torch.rand(32, 100, 4)
    targ = torch.rand(32, 100, 4)
    loss = cylinder_loss_reg(inputs,targ)
    print(loss)
