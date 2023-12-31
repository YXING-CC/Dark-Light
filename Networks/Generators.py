import torch
import torch.nn as nn
from Networks.Transformer_Encoder import TransformerEncoderLayer, TransformerEncoder
from Networks.Transformer_decoder import TransformerDecoderLayer, TransformerDecoder
from Networks.FiLM import Model as Model_Film
from Networks.Informer import Model
from Networks.wavenet import TemporalConvNet

from Film_configs import Configs_pred, Configs_recons, Configs_pred_autoformer, Configs_recons_autoformer

import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dark_gen_pred_lstm(nn.Module):
    def __init__(self, feat_dim = 4, device=None):
        super(dark_gen_pred_lstm, self).__init__()
        feat_dim = 4
        hid_dim = 128

        self.lstm_pred = nn.LSTM(feat_dim, hid_dim, 2, batch_first=True, bidirectional=True)
        self.linear_pred_red = nn.Linear(100*256, 50*256)

    def forward(self, x, targ, recons_out):
        dec_out, _ = self.lstm_pred(x)
        enc_out = dec_out + recons_out

        dec_size = dec_out.size()
        dec_out = dec_out.reshape(dec_size[0], -1)
        dec_out = self.linear_pred_red(dec_out)
        dec_out = dec_out.reshape(dec_size[0], 50, 256)

        return dec_out

class dark_gen_recons_lstm(nn.Module):
    def __init__(self, feat_dim=4, device=None):
        super(dark_gen_recons_lstm, self).__init__()
        feat_dim = 4
        hid_dim = 128

        self.lstm_recons = nn.LSTM(feat_dim, hid_dim, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        recons_out, _ = self.lstm_recons(x)
        return recons_out

#------------------------TCN-------------------------------------------
class dark_gen_pred_tcn(nn.Module):
    def __init__(self, feat_dim = 4, device=None):
        super(dark_gen_pred_tcn, self).__init__()

        num_chans = [100] * (5 - 1) + [100]
        self.tcn_pred = TemporalConvNet(num_inputs=100, num_channels=num_chans)

        self.linear_pred_red = nn.Linear(100*256, 50*256)
        self.linear1 = nn.Linear(4,256)

    def forward(self, x, targ, recons_out):
        dec_out = self.tcn_pred(x)
        dec_out = self.linear1(dec_out)
        enc_out = dec_out + recons_out

        dec_size = dec_out.size()
        dec_out = dec_out.reshape(dec_size[0], -1)
        dec_out = self.linear_pred_red(dec_out)
        dec_out = dec_out.reshape(dec_size[0], 50, 256)

        return dec_out

class dark_gen_recons_tcn(nn.Module):
    def __init__(self, feat_dim=4, device=None):
        super(dark_gen_recons_tcn, self).__init__()

        num_chans = [100] * (5 - 1) + [100]
        self.tcn_recons = TemporalConvNet(num_inputs=100, num_channels=num_chans)
        self.linear2 = nn.Linear(4,256)

    def forward(self, x):
        recons_out = self.tcn_recons(x)
        recons_out = self.linear2(recons_out)

        return recons_out


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


class dark_gen_pred(nn.Module):
    def __init__(self, feat_dim = 4, device=None):
        super(dark_gen_pred, self).__init__()

        self.hid_dim = 256
        num_layers = 6

        trans_dpt = 0.05

        self.encoder_layer = TransformerEncoderLayer(d_model=self.hid_dim, nhead=8, dropout=trans_dpt, batch_first='True')
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers * 2)

        self.pe = PositionalEncoder(d_model=self.hid_dim)
        self.linear_emb = nn.Linear(feat_dim, self.hid_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        #
        self.mask = self.generate_square_subsequent_mask(50, device)
        # print('mask', self.mask)
        self.decoder = nn.Linear(100*self.hid_dim, 50*self.hid_dim)

    def forward(self, x, targ, recons_out):
        x = self.linear_emb(x)
        x = self.pe(x)

        enc_out = self.transformer_encoder(x)
        # print('enc_out.size()', enc_out.size())
        enc_out = enc_out + recons_out

        dec_out = enc_out.reshape(enc_out.size()[0], -1)
        dec_out = self.decoder(dec_out)
        dec_out = dec_out.reshape(dec_out.size()[0], 50, self.hid_dim)

        return dec_out

class dark_gen_recons(nn.Module):
    def __init__(self, feat_dim=4, device=None):
        super(dark_gen_recons, self).__init__()

        hid_dim = 256
        num_layers = 6
        trans_dpt = 0.05

        self.encoder_layer_recons = TransformerEncoderLayer(d_model=hid_dim, nhead=8, dropout=trans_dpt, batch_first='True')
        self.transformer_encoder_recons = TransformerEncoder(self.encoder_layer_recons, num_layers=num_layers * 2)
        
        self.pe = PositionalEncoder(d_model=hid_dim)
        self.linear_emb = nn.Linear(feat_dim, hid_dim)

    def forward(self, x):
        x = self.linear_emb(x)
        x = self.pe(x)

        recons_out = self.transformer_encoder_recons(x)

        return recons_out
        

########################Informer#############################
class dark_gen_pred_informer(nn.Module):
    def __init__(self, args, feat_dim=4, device=None):
        super(dark_gen_pred_autoformer, self).__init__()
        hid_dim = 256
        feat_dim1 = 32
        self.linear_emb = nn.Linear(feat_dim, feat_dim1)

        self.model = Model(args)
        self.linear = nn.Linear(feat_dim, hid_dim)

    def forward(self, x, x_mark, targ, targ_mark):
        dec_out, _ = self.model(x, x_mark, targ, targ_mark)
        dec_out = self.linear(dec_out)

        return dec_out


class dark_gen_recons_informer(nn.Module):
    def __init__(self, args, feat_dim=4, device=None):
        super(dark_gen_recons_autoformer, self).__init__()
        hid_dim = 256
        feat_dim1 = 32

        self.linear_emb = nn.Linear(feat_dim, feat_dim1)
        self.model = Model(args)
        self.linear = nn.Linear(feat_dim, hid_dim)

    def forward(self, x, x_mark, targ, targ_mark):
        recons_out, _ = self.model(x, x_mark, targ, targ_mark)
        recons_out = self.linear(recons_out)

        # print('recons_out here', recons_out.size())

        return recons_out

########################Informer_end#############################

class pred_head(nn.Module):
    def __init__(self, num_fault=9, device=None):
        super(pred_head, self).__init__()

        hid_dim = 256
        feat_dim = 4

        self.relu = nn.ReLU()
        self.linear_regress = nn.Linear(hid_dim, feat_dim)
        self.linear_recons = nn.Linear(hid_dim, feat_dim)

        self.fault_type = num_fault

        self.dropout = nn.Dropout(p=0.1)
        self.dropout1 = nn.Dropout(p=0.1)

        self.linear_sig1 = nn.Linear(hid_dim*150, self.fault_type)
        self.linear_sig2 = nn.Linear(hid_dim*150, self.fault_type)
        self.linear_sig3 = nn.Linear(hid_dim*150, self.fault_type)
        self.linear_sig4 = nn.Linear(hid_dim*150, self.fault_type)

    def forward(self, dec_out, recons_out):
        # print('dec_out, recons_out', dec_out.size(), recons_out.size())
        reg_output = self.linear_regress(self.dropout(dec_out))
        recons_output = self.linear_recons(self.dropout(recons_out))

        dec_out = torch.cat([dec_out, recons_out], dim=1)

        # print('dec_out', dec_out.size())

        dec_size = dec_out.size()
        dec_out = dec_out.reshape(dec_size[0], -1)

        fault_sig_1 = self.linear_sig1(self.dropout1(dec_out))
        fault_sig_2 = self.linear_sig2(self.dropout1(dec_out))
        fault_sig_3 = self.linear_sig3(self.dropout1(dec_out))
        fault_sig_4 = self.linear_sig4(self.dropout1(dec_out))

        return reg_output, recons_output, [fault_sig_1, fault_sig_2, fault_sig_3, fault_sig_4]


if __name__ == "__main__":
    inputs = torch.rand(32, 100, 4)
    targ = torch.rand(32, 50, 4)
    # net = dark_gen_pred()
    # net_re = dark_gen_recons()

    net = dark_gen_pred()
    net_re = dark_gen_recons()

