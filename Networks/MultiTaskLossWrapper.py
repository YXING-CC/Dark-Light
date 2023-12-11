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


if __name__ == "__main__":
    inputs = torch.rand(32, 100, 4)
    targ = torch.rand(32, 100, 4)
    loss = cylinder_loss_reg(inputs,targ)
    print(loss)
