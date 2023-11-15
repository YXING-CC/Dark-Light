import torch
import torch.nn as nn
from Networks.Transformer_Encoder import TransformerEncoderLayer, TransformerEncoder
from Networks.Transformer_decoder import TransformerDecoderLayer, TransformerDecoder

import math
from Networks.Informer import Model
from Film_configs import Configs_pred, Configs_recons, Configs_pred_autoformer, Configs_recons_autoformer
from Networks.wavenet import TemporalConvNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class dark_net_tcn(nn.Module):
    def __init__(self, device=None):
        super(dark_net_tcn, self).__init__()

        num_chans = [100] * (5 - 1) + [100]

        self.tcn_pred = TemporalConvNet(num_inputs=100, num_channels=num_chans)
        self.tcn_recons = TemporalConvNet(num_inputs=100, num_channels=num_chans)

        self.linear_pred_red = nn.Linear(100*4, 50*4)

        self.linear1 = nn.Linear(4,256)
        self.linear2 = nn.Linear(4,256)

    def forward(self, x, targ):
        dec_out = self.tcn_pred(x)
        recons_out = self.tcn_recons(x)

        dec_size = dec_out.size()
        # print('dec_out.size', dec_size)

        dec_out = dec_out.reshape(dec_size[0], -1)
        # print('dec_out.size', dec_out.size())

        dec_out = self.linear_pred_red(dec_out)
        dec_out = dec_out.reshape(dec_size[0], 50, 4)

        dec_out = self.linear1(dec_out)
        recons_out = self.linear2(recons_out)

        return dec_out, recons_out

class dark_net_LSTM(nn.Module):
    def __init__(self, device=None):
        super(dark_net_LSTM, self).__init__()
        feat_dim = 4
        hid_dim = 128
        num_layers = 3

        trans_drop = 0.3

        self.lstm_pred = nn.LSTM(feat_dim, hid_dim, 2, batch_first=True, bidirectional=True)
        self.lstm_recons = nn.LSTM(feat_dim, hid_dim, 2, batch_first=True, bidirectional=True)

        self.gru_pred = nn.GRU(feat_dim, hid_dim, 2, batch_first=True, bidirectional=True)
        self.gru_recons = nn.GRU(feat_dim, hid_dim, 2, batch_first=True, bidirectional=True)

        self.linear_emb_targ = nn.Linear(feat_dim, hid_dim)
        self.linear_pred_red = nn.Linear(100*256, 50*256)

    def forward(self, x, targ):

        dec_out, _ = self.gru_pred(x)
        recons_out, _ = self.gru_recons(x)

        dec_size = dec_out.size()
        # print(dec_size)
        dec_out = dec_out.reshape(dec_size[0], -1)
        dec_out = self.linear_pred_red(dec_out)
        dec_out = dec_out.reshape(dec_size[0], 50, 256)
        # print('LSTM dec', dec_out.size(), 'Recons', recons_out.size())

        return dec_out, recons_out


class dark_net(nn.Module):
    def __init__(self, device=None):
        super(dark_net, self).__init__()
        feat_dim = 4
        self.hid_dim = 256
        num_layers = 3

        trans_drop = 0.15

        self.encoder_layer = TransformerEncoderLayer(d_model=self.hid_dim, nhead=8, dropout=trans_drop, batch_first='True')
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers*2)

        # self.encoder_layer_2 = TransformerEncoderLayer(d_model=hid_dim, nhead=8, dropout=trans_drop, batch_first='True')
        # self.transformer_encoder_2 = TransformerEncoder(self.encoder_layer_2, num_layers=num_layers*1)

        self.encoder_layer_recons = TransformerEncoderLayer(d_model=self.hid_dim, nhead=8, dropout=trans_drop, batch_first='True')
        self.transformer_encoder_recons = TransformerEncoder(self.encoder_layer_recons, num_layers=num_layers*2)
    #
        self.decoder_layer = TransformerDecoderLayer(d_model=self.hid_dim, nhead=8, dropout=trans_drop, batch_first=True)
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.pe = PositionalEncoder(d_model=self.hid_dim)
        self.linear_emb = nn.Linear(feat_dim, self.hid_dim)

        self.pe_targ = PositionalEncoder(d_model=self.hid_dim)
        self.linear_emb_targ = nn.Linear(feat_dim, self.hid_dim)
        #
        self.mask = self.generate_square_subsequent_mask(50, device)
        # print('mask', self.mask)
        self.decoder = nn.Linear(100*self.hid_dim, 50*self.hid_dim)

    def forward(self, x, targ):
        x = self.linear_emb(x)
        x = self.pe(x)
        # print('output.size:', output.size())
        recons_out = self.transformer_encoder_recons(x)

        enc_out = self.transformer_encoder(x)
        # enc_out = self.transformer_encoder_2(x+recons_out)

        dec_out = enc_out.reshape(enc_out.size()[0], -1)
        dec_out = self.decoder(dec_out)
        dec_out = dec_out.reshape(dec_out.size()[0], 50, self.hid_dim)

        # targ = self.linear_emb_targ(targ)
        # targ = self.pe_targ(targ)
        #
        # dec_out = self.transformer_decoder(targ, enc_out+recons_out, tgt_mask = self.mask)

        # print('dec_out.size()', dec_out.size(), 'recons_out.size()', recons_out.size())

        return dec_out, recons_out
        # return output

    def generate_square_subsequent_mask(self, sz: int, device='cpu'):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'),  device=device), diagonal=1)

########################Informer_start#############################
class dark_net_informer(nn.Module):
    def __init__(self, args, args_recons, feat_dim=4, device=None):
        super(dark_net_informer, self).__init__()
        hid_dim = 256
        feat_dim1 = 32
        self.model_pred = Model(args)
        self.linear_pred = nn.Linear(feat_dim, hid_dim)

        self.model_recons = Model(args_recons)
        self.linear_recons = nn.Linear(feat_dim, hid_dim)

    def forward(self, x, x_mark, targ, targ_mark):
        dec_out, _ = self.model_pred(x, x, targ, targ)
        dec_out = self.linear_pred(dec_out)

        recons_out, _ = self.model_recons(x, x, x, x)
        recons_out = self.linear_recons(recons_out)

        return dec_out, recons_out
########################Informer_end#############################

class pred_head(nn.Module):
    def __init__(self, num_fault=5, device=None):
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
        # print('dec_out', dec_out.size(), 'recons_out()', recons_out.size())
        reg_output = self.linear_regress(self.dropout(dec_out))
        recons_output = self.linear_recons(self.dropout(recons_out))

        dec_out = torch.cat([dec_out, recons_out], dim=1)

        dec_size = dec_out.size()
        # print('dec_size', dec_size)
        dec_out = dec_out.reshape(dec_size[0], -1)

        # print('dec_out.size()', dec_out.size())
        #
        fault_sig_1 = self.linear_sig1(self.dropout1(dec_out))
        fault_sig_2 = self.linear_sig2(self.dropout1(dec_out))
        fault_sig_3 = self.linear_sig3(self.dropout1(dec_out))
        fault_sig_4 = self.linear_sig4(self.dropout1(dec_out))

        return reg_output, recons_output, [fault_sig_1, fault_sig_2, fault_sig_3, fault_sig_4]


if __name__ == "__main__":
    inputs = torch.rand(32, 100, 4)
    targ = torch.rand(32, 50, 4)
    # net = dark_net_LSTM()
    # net = dark_net_informer(Configs_pred_autoformer, Configs_recons_autoformer)
    net = dark_net_tcn()

    pred_head = pred_head(num_fault=9)
    # reg, recons = net.forward(inputs, inputs, targ, targ)

    reg, recons = net.forward(inputs, targ)
    print('reg.size1:', reg.size(), 'recons.size1:', recons.size())

    reg_pred, recons_pred, cls = pred_head.forward(reg, recons)
    print('reg.size:', reg_pred.size(), 'recons.size:', recons_pred.size(), 'cls.size', cls[1].size())
