import torch
import torch.nn as nn
from Networks.Transformer_Encoder import TransformerEncoderLayer, TransformerEncoder
from Networks.Transformer_decoder import TransformerDecoderLayer, TransformerDecoder

import math


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=100):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x


class TF_FD_14dim(nn.Module):
    def __init__(self, num_fault=5, device = None):
        super(TF_FD_14dim, self).__init__()
        feat_dim = 4
        hid_dim = 256
        num_layers = 6
        self.fault_type = num_fault

        self.encoder_layer = TransformerEncoderLayer(d_model=hid_dim, nhead=8, batch_first='True')
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers*1)
        
        self.pe = PositionalEncoder(d_model=hid_dim)
        self.linear_emb = nn.Linear(feat_dim, hid_dim)

        self.relu = nn.ReLU()
        self.linear_regress = nn.Linear(hid_dim, feat_dim)

        self.dropout = nn.Dropout(p=0.2)

        self.linear_sig1 = nn.Linear(hid_dim*50, self.fault_type)
        self.linear_sig2 = nn.Linear(hid_dim*50, self.fault_type)
        self.linear_sig3 = nn.Linear(hid_dim*50, self.fault_type)
        self.linear_sig4 = nn.Linear(hid_dim*50, self.fault_type)

        self.sftmax = nn.Softmax()
        self.m = nn.Conv1d(100, 50, 3, stride=2)
        self.n = nn.Linear(127,256)

    def forward(self, x, targ, pred_feat, recons_feat):
        # print('input_t', x.size(), targ.size())
        x = self.linear_emb(x)
        x = self.pe(x)
        enc_out = self.transformer_encoder(x)

        # print('enc_out.size:', enc_out.size())
        # recons_feat = (recons_feat + self.recons_multiattn(recons_feat, recons_feat, recons_feat)[0])
        # pred_feat = (pred_feat + self.pred_multiattn(pred_feat, pred_feat, pred_feat)[0])

        enc_out = enc_out + recons_feat
        dec_out = self.n(self.m(enc_out))

        dec_out = dec_out + pred_feat

        reg_output = self.linear_regress(self.dropout(dec_out))

        dec_size = dec_out.size()
        dec_out = dec_out.reshape(dec_size[0], -1)

        # print('dec_out.size()', dec_out.size())

        fault_sig_1 = self.linear_sig1(dec_out)
        fault_sig_2 = self.linear_sig2(dec_out)
        fault_sig_3 = self.linear_sig3(dec_out)
        fault_sig_4 = self.linear_sig4(dec_out)

        return reg_output, [fault_sig_1, fault_sig_2, fault_sig_3, fault_sig_4]
        # return output


if __name__ == "__main__":
    inputs = torch.rand(32, 100, 4)
    targ = torch.rand(32, 50, 4)

    recons = torch.rand(32, 100, 256)
    pred = torch.rand(32, 50, 256)

    net = TF_FD_14dim(num_fault=9)
    reg, cls = net.forward(inputs, targ, pred, recons)
    print('reg.size:', reg.size(), 'cls.size', cls[1].size())

    model = net
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size ) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))


