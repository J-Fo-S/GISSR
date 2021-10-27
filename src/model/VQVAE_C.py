import torch
from torch import nn
from torch.nn import functional as F


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x, y=None):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, (1, 3), padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1, padding=0),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, (1, 4), stride=2, padding=(0,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, (1, 4), stride=2, padding=(0,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, (1, 3), padding=(0,1)),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, (1, 4), stride=2, padding=(0,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, (1, 3), padding=(0,1)),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, (1, 3), padding=(0,1))]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, (1, 4), stride=2, padding=(0,1)),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, (1, 4), stride=2, padding=(0,1)
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, (1, 4), stride=2, padding=(0,1))
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE_C(nn.Module):
    # Conversions to figure out: in_channel to 1? padding for what? filters to 3x1? DAE_C feature_dim pointless - only for FCN?
    # keep bottom/top encoders and add decoder?
    # TO-DO: add model_dict and args
    def __init__(
        self,
        model_dict=None,
        args=None,
        logger=None,
        in_channel=1,
        channel=32,
        n_res_block=2,
        n_res_channel=16,
        embed_dim=32,
        n_embed=64,
        decay=0.99,
    ):
        super().__init__()
        self.args = args
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, (1, 4), stride=2, padding=(0,1)
        )
        if self.args.data_feature=="lps_lts":
            self.enc_bc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
            self.quantize_conv_bc = nn.Conv2d(channel, embed_dim, 1)
            self.quantize_bc = Quantize(embed_dim, n_embed)
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input_0, input_1):
        #if self.args.data_feature=="lps_lts":
        #    _, quant_b_lps, diff_lps, _, _ = self.encode(input[0])
        #    quant_t_lts, _, diff_lts, _, _ = self.encode(input[1])
        #    dec = self.decode(quant_t_lts, quant_b_lps)
        #    diff = diff_lps + diff_lts
        #else:
        if self.args.data_feature=="lps_lts":
            quant_t, quant_b, diff, _, _, _ = self.encode_c(input_1, input_0)
        else:
            quant_t, quant_b, diff, _, _ = self.encode(input_0)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input_0):
        enc_b = self.enc_b(input_0)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b
    
    def encode_c(self, input_0, input_1):
        enc_b = self.enc_b(input_0)
        enc_bc = self.enc_bc(input_1)
        enc_t = self.enc_t(enc_bc)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        quant_bc = self.quantize_conv_bc(enc_bc).permute(0, 2, 3, 1)
        quant_bc, diff_bc, id_bc = self.quantize_bc(quant_bc)
        quant_bc = quant_bc.permute(0, 3, 1, 2)
        diff_bc = diff_bc.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b + diff_bc, id_t, id_b, id_bc

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec