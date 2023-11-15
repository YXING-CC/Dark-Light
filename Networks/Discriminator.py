import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, Transformer
import torch.nn.functional as F
import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print("Device being used:", device)

class Conv1d_tq_discriminator(nn.Module):
    def __init__(self, num_layers=3, input_dim=64, output_dim=64, pred_len=50):
        super(Conv1d_tq_discriminator, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, output_dim, output_dim)) for i in range(num_layers)])
        self.conv_1x1_last = nn.Conv1d(output_dim, 1, kernel_size=5)

        self.leakrelu = nn.LeakyReLU(0.2, inplace=False)
        self.linear = nn.Linear(pred_len - 4, 1)

        self.lstm_dim = 32
        self.lstm_pre = nn.LSTM(input_dim, self.lstm_dim, num_layers=3, batch_first=True, bidirectional=True)


    def forward(self, x):
        x = torch.transpose(x, 1, 2)

        x = self.conv_1x1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.conv_1x1_last(x)
        x = torch.squeeze(x)
        # print('x.size',x.size())
        x = self.linear(self.leakrelu(x))

        return x

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.leaky_relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


if __name__ == "__main__":
    inputs = torch.rand(32, 50, 4)
    net = Conv1d_tq_discriminator(num_layers=3, input_dim=4, output_dim=4, pred_len=50)
    out = net(inputs)
    print('out.size:', out.size())