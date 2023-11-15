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
        num_layers = 3
        self.fault_type = num_fault

        self.encoder_layer = TransformerEncoderLayer(d_model=hid_dim, nhead=8, batch_first='True')
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers*2)

        self.decoder_layer = TransformerDecoderLayer(d_model=hid_dim, nhead=8, batch_first=True)
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.pe = PositionalEncoder(d_model=hid_dim)
        self.linear_emb = nn.Linear(feat_dim, hid_dim)

        self.pe_targ = PositionalEncoder(d_model=hid_dim)
        self.linear_emb_targ = nn.Linear(feat_dim, hid_dim)

        self.relu = nn.ReLU()
        self.linear_regress = nn.Linear(hid_dim, feat_dim)

        self.dropout = nn.Dropout(p=0.2)

        self.linear_sig1 = nn.Linear(hid_dim*50, self.fault_type)
        self.linear_sig2 = nn.Linear(hid_dim*50, self.fault_type)
        self.linear_sig3 = nn.Linear(hid_dim*50, self.fault_type)
        self.linear_sig4 = nn.Linear(hid_dim*50, self.fault_type)

        self.mask = self.generate_square_subsequent_mask(50, device)

        self.sftmax = nn.Softmax()

        self.layernorm_recons = nn.LayerNorm(hid_dim, eps=1e-05)
        self.layernorm_pred = nn.LayerNorm(hid_dim, eps=1e-05)

        self.recons_multiattn = nn.MultiheadAttention(hid_dim, 8)
        self.pred_multiattn = nn.MultiheadAttention(hid_dim, 8)
        self.decoder = nn.Linear(100*hid_dim, 50*hid_dim)


    def forward(self, x, targ, pred_feat, recons_feat):
        # print('input_t', x.size(), targ.size())
        x = self.linear_emb(x)
        x = self.pe(x)
        enc_out = self.transformer_encoder(x)

        # print('enc_out.size:', enc_out.size())
        # recons_feat = (recons_feat + self.recons_multiattn(recons_feat, recons_feat, recons_feat)[0])
        # pred_feat = (pred_feat + self.pred_multiattn(pred_feat, pred_feat, pred_feat)[0])

        enc_out = enc_out + recons_feat

        dec_out = enc_out.reshape(enc_out.size()[0], -1)
        dec_out = self.decoder(dec_out)
        dec_out = dec_out.reshape(dec_out.size()[0], 50, 256)
        #
        # targ = self.linear_emb_targ(targ)
        # # print('targ.size',targ.size())
        # targ = self.pe_targ(targ)
        #
        # targ = targ
        #
        # dec_out = self.transformer_decoder(targ, enc_out, tgt_mask=self.mask)
        #
        # # print('dec_out.size()', dec_out.size())
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

    def generate_square_subsequent_mask(self, sz: int, device='cpu'):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'),  device=device), diagonal=1)


if __name__ == "__main__":
    inputs = torch.rand(32, 100, 4)
    targ = torch.rand(32, 50, 4)

    recons = torch.rand(32, 100, 256)
    pred = torch.rand(32, 50, 256)

    net = TF_FD_14dim(num_fault=5)
    reg, cls = net.forward(inputs, targ, pred, recons)

    print('reg.size:', reg.size(), 'cls.size', cls[1].size())

