import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools

from .module_util import (
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, Residual, Identity, EnhancedCrossAttentionFusion)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ch=64, ch_mult=[1, 2, 4, 4], embed_dim=4):
        super().__init__()
        self.depth = len(ch_mult)

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_ch, ch, 3)

        # layers
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        ch_mult = [1] + ch_mult
        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i+1]
            self.encoder.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in),
                block_class(dim_in=dim_in, dim_out=dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if i == (self.depth-1) else Identity(),
                Downsample(dim_in, dim_out) if i != (self.depth-1) else default_conv(dim_in, dim_out)
            ]))

            self.decoder.insert(0, nn.ModuleList([
                EnhancedCrossAttentionFusion(in_dim_x=dim_in, in_dim_y=dim_in, out_dim=dim_in),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out),
                EnhancedCrossAttentionFusion(in_dim_x=dim_in, in_dim_y=dim_in, out_dim=dim_in),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if i == (self.depth-1) else Identity(),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = ch * ch_mult[-1]
        self.latent_conv = default_conv(mid_dim, embed_dim, 1)
        self.post_latent_conv = default_conv(embed_dim, mid_dim, 1)
        # self.final_at = ResidualCrossAttention1(in_channels_1=ch, in_channels_2=ch, out_channels=ch)
        self.final_at = EnhancedCrossAttentionFusion(in_dim_x=ch, in_dim_y=ch, out_dim=ch)
        self.final_conv = nn.Conv2d(ch, out_ch, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def encode(self, x):
        self.H, self.W = x.shape[2:]
        x = self.check_image_size(x, self.H, self.W)

        x = self.init_conv(x)
        h = [x]
        for b1, b2, attn, downsample in self.encoder:
            x = b1(x)
            h.append(x)

            x = b2(x)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.latent_conv(x)
        return x, h

    def decode(self, x, h1, h2):
        x = self.post_latent_conv(x)
        for i, (r1, b1, r2, b2, attn, upsample) in enumerate(self.decoder):
            # h11 = torch.cat([h1[-(i*2+1)], h2[-(i*2+1)]], dim=1)
            h11 = r1(h1[-(i*2+1)], h2[-(i*2+1)])
            x = torch.cat([x, h11], dim=1)
            x = b1(x)
            # h12 = torch.cat([h1[-(i*2+2)], h2[-(i*2+2)]], dim=1)
            h12 = r2(h1[-(i*2+2)], h2[-(i*2+2)])
            x = torch.cat([x, h12], dim=1)
            x = b2(x)
            x = attn(x)
            x = upsample(x)
        a1 = self.final_at(h1[0], h2[0])
        x = self.final_conv(x + a1)
        # x = self.final_conv(x + h1[0] + h2[0])
        return x[..., :self.H, :self.W]

    def forward(self, x):
        x, h = self.encode(x)
        x = self.decode(x, h)

        return x



