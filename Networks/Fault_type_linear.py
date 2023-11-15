import torch
import torch.nn as nn

in_dim_x = 100

class Cylinder_fault_linear(nn.Module):
    def __init__(self, num_fault=5):
        super(Cylinder_fault_linear, self).__init__()
        in_dim = in_dim_x
        out_dim = num_fault
        self.linear_cylind_sig1 = nn.Linear(in_dim, out_dim)
        self.linear_cylind_sig2 = nn.Linear(in_dim, out_dim)
        self.linear_cylind_sig3 = nn.Linear(in_dim, out_dim)
        self.linear_cylind_sig4 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        cylind_fault_sig1 = self.linear_cylind_sig1(x[:, :, 0])
        cylind_fault_sig2 = self.linear_cylind_sig2(x[:, :, 1])
        cylind_fault_sig3 = self.linear_cylind_sig3(x[:, :, 2])
        cylind_fault_sig4 = self.linear_cylind_sig4(x[:, :, 3])

        return [cylind_fault_sig1, cylind_fault_sig2, cylind_fault_sig3, cylind_fault_sig4]

class CAN_fault_linear(nn.Module):
    def __init__(self, num_fault=5):
        super(CAN_fault_linear, self).__init__()
        in_dim = in_dim_x
        out_dim = num_fault
        self.linear_can_sig1 = nn.Linear(in_dim, out_dim)
        self.linear_can_sig2 = nn.Linear(in_dim, out_dim)
        self.linear_can_sig3 = nn.Linear(in_dim, out_dim)
        self.linear_can_sig4 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        can_fault_sig1 = self.linear_can_sig1(x[:, :, 0])
        can_fault_sig2 = self.linear_can_sig2(x[:, :, 1])
        can_fault_sig3 = self.linear_can_sig3(x[:, :, 2])
        can_fault_sig4 = self.linear_can_sig4(x[:, :, 3])

        return [can_fault_sig1, can_fault_sig2, can_fault_sig3, can_fault_sig4]


class Motor_fault_linear(nn.Module):
    def __init__(self, num_fault=5):
        super(Motor_fault_linear, self).__init__()
        in_dim = in_dim_x
        out_dim = num_fault
        self.linear_motor_sig1 = nn.Linear(in_dim, out_dim)
        self.linear_motor_sig2 = nn.Linear(in_dim, out_dim)
        self.linear_motor_sig3 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        motor_fault_sig1 = self.linear_motor_sig1(x[:, :, 0])
        motor_fault_sig2 = self.linear_motor_sig2(x[:, :, 1])
        motor_fault_sig3 = self.linear_motor_sig3(x[:, :, 2])

        return [motor_fault_sig1, motor_fault_sig2, motor_fault_sig3]


class Battery_fault_linear(nn.Module):
    def __init__(self, num_fault=5):
        super(Battery_fault_linear, self).__init__()
        in_dim = in_dim_x
        out_dim = num_fault
        self.linear_battery_sig1 = nn.Linear(in_dim, out_dim)
        self.linear_battery_sig2 = nn.Linear(in_dim, out_dim)
        self.linear_battery_sig3 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        battery_fault_sig1 = self.linear_battery_sig1(x[:, :, 0])
        battery_fault_sig2 = self.linear_battery_sig2(x[:, :, 1])
        battery_fault_sig3 = self.linear_battery_sig3(x[:, :, 2])

        return [battery_fault_sig1, battery_fault_sig2, battery_fault_sig3]


if __name__ == "__main__":
    inputs = torch.rand(32, 100, 4)
    net = Cylinder_fault_linear(num_fault=5)
    sig_info = net.forward(inputs)
    print('sig1.size:', sig_info[0].size(), 'sig4.size:', sig_info[3].size())