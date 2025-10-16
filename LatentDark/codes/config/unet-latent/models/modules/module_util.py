import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch import einsum

from einops import rearrange, reduce
from einops.layers.torch import Rearrange


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


def NonLinearity(inplace=False):
    return nn.SiLU(inplace)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def Normalize1(in_channels):
    return nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6, affine=True)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, 1, 1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


def default_conv(dim_in, dim_out, kernel_size=3, bias=False):
    return nn.Conv2d(dim_in, dim_out, kernel_size, padding=(kernel_size//2), bias=bias)


class Block(nn.Module):
    def __init__(self, conv, dim_in, dim_out, act=NonLinearity()):
        super().__init__()
        self.proj = conv(dim_in, dim_out)
        self.act = act

    def forward(self, x, scale_shift=None):
        x = self.proj(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, conv, dim_in, dim_out, time_emb_dim=None, act=NonLinearity()):
        super(ResBlock, self).__init__()
        self.mlp = nn.Sequential(
            act, nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim else None

        self.block1 = Block(conv, dim_in, dim_out, act)
        self.block2 = Block(conv, dim_out, dim_out, act)
        self.res_conv = conv(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


# channel attention
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


# self attention on each channel
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def initialize_weights(net_l, scale=1.):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

# 空间的交叉注意力
class ResidualCrossAttention(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, kernel_size=3, max_size=128):
        super(ResidualCrossAttention, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels_1)
        self.norm2 = nn.BatchNorm2d(in_channels_2)

        self.q_conv1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=1)
        self.k_conv1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=1)
        self.v_conv1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=1)

        self.q_conv2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=1)
        self.k_conv2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=1)
        self.v_conv2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=1)

        self.depthwise_conv1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=kernel_size, groups=in_channels_1,
                                         padding=kernel_size // 2)
        self.depthwise_conv2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=kernel_size, groups=in_channels_2,
                                         padding=kernel_size // 2)

        self.linear_proj = nn.Conv2d(in_channels_1 + in_channels_2, out_channels, kernel_size=1)

        self.residual_conv1 = nn.Conv2d(in_channels_1, out_channels, kernel_size=1)
        self.residual_conv2 = nn.Conv2d(in_channels_2, out_channels, kernel_size=1)

        self.max_size = max_size
        self.downsample = Downsample(in_channels_1, in_channels_1)
        self.upsample = Upsample(out_channels, out_channels)

    def forward(self, F_i, F_w):
        B, C1, H, W = F_i.shape
        _, C2, _, _ = F_w.shape

        # Determine the number of downsampling steps based on input size
        downsample_steps = 0
        if H >= 400 or W >= 400:
            downsample_steps = 3
        elif H >= 200 or W >= 200:
            downsample_steps = 2
        elif H >= 100 or W >= 100:
            downsample_steps = 1

        # Apply downsampling
        for _ in range(downsample_steps):
            F_i = self.downsample(F_i)
            F_w = self.downsample(F_w)

        _, _, H, W = F_i.shape

        # Normalization
        F_i_norm = self.norm1(F_i)
        F_w_norm = self.norm2(F_w)

        # Depth-wise Convolution
        F_i_dw = self.depthwise_conv1(F_i_norm)
        F_w_dw = self.depthwise_conv2(F_w_norm)

        # Point-wise Convolution (Query, Key, Value)
        Q_i = self.q_conv1(F_i_dw).view(B, -1, H * W)  # Shape: (B, C1, HW)
        K_i = self.k_conv1(F_i_dw).view(B, -1, H * W)  # Shape: (B, C1, HW)
        V_i = self.v_conv1(F_i_dw).view(B, -1, H * W)  # Shape: (B, C1, HW)

        Q_w = self.q_conv2(F_w_dw).view(B, -1, H * W)  # Shape: (B, C2, HW)
        K_w = self.k_conv2(F_w_dw).view(B, -1, H * W)  # Shape: (B, C2, HW)
        V_w = self.v_conv2(F_w_dw).view(B, -1, H * W)  # Shape: (B, C2, HW)

        # Attention Mechanism
        attention_1 = F.softmax(torch.bmm(Q_i.transpose(1, 2), K_w), dim=-1)  # Shape: (B, HW, HW)
        attention_2 = F.softmax(torch.bmm(Q_w.transpose(1, 2), K_i), dim=-1)  # Shape: (B, HW, HW)

        A_i = torch.bmm(V_w, attention_1).view(B, C2, H, W)  # Shape: (B, C2, H, W)
        A_w = torch.bmm(V_i, attention_2).view(B, C1, H, W)  # Shape: (B, C1, H, W)

        # Concatenate and Linear Projection
        concat = torch.cat((A_i, A_w), dim=1)
        output = self.linear_proj(concat)

        # Residual Connection with channel matching
        F_i_res = self.residual_conv1(F_i)
        F_w_res = self.residual_conv2(F_w)

        output = output + F_i_res + F_w_res

        # Upsample to original size if downsampled
        for _ in range(downsample_steps):
            output = self.upsample(output)

        return output

# 通道的交叉注意力
class ResidualCrossAttention1(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, kernel_size=3):
        super(ResidualCrossAttention1, self).__init__()
        self.norm1 = LayerNorm(in_channels_1)
        self.norm2 = LayerNorm(in_channels_2)

        # self.norm1 = nn.BatchNorm2d(in_channels_1)
        # self.norm2 = nn.BatchNorm2d(in_channels_2)

        # self.norm1 = Normalize1(in_channels_1)
        # self.norm2 = Normalize1(in_channels_2)

        self.q_conv1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=1)
        self.k_conv1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=1)
        self.v_conv1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=1)

        self.q_conv2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=1)
        self.k_conv2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=1)
        self.v_conv2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=1)

        self.q_dw_conv1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=kernel_size, groups=in_channels_1,
                                    padding=kernel_size // 2)
        self.k_dw_conv1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=kernel_size, groups=in_channels_1,
                                    padding=kernel_size // 2)
        self.v_dw_conv1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=kernel_size, groups=in_channels_1,
                                    padding=kernel_size // 2)

        self.q_dw_conv2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=kernel_size, groups=in_channels_2,
                                    padding=kernel_size // 2)
        self.k_dw_conv2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=kernel_size, groups=in_channels_2,
                                    padding=kernel_size // 2)
        self.v_dw_conv2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=kernel_size, groups=in_channels_2,
                                    padding=kernel_size // 2)

        # self.depthwise_conv1 = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=kernel_size, groups=in_channels_1,
        #                                  padding=kernel_size // 2)
        # self.depthwise_conv2 = nn.Conv2d(in_channels_2, in_channels_2, kernel_size=kernel_size, groups=in_channels_2,
        #                                  padding=kernel_size // 2)

        # self.linear_proj = nn.Conv2d(in_channels_1 + in_channels_2, out_channels, kernel_size=1)
        self.linear_proj = nn.Linear(in_channels_1 + in_channels_2, out_channels)

        # self.residual_conv1 = nn.Conv2d(in_channels_1, out_channels, kernel_size=1)
        # self.residual_conv2 = nn.Conv2d(in_channels_2, out_channels, kernel_size=1)

    def forward(self, F_i, F_w):
        B, C1, H, W = F_i.shape
        _, C2, _, _ = F_w.shape

        # Normalization
        F_i_dw = self.norm1(F_i)
        F_w_dw = self.norm2(F_w)

        # Depth-wise Convolution
        # F_i_dw = self.depthwise_conv1(F_i_norm)
        # F_w_dw = self.depthwise_conv2(F_w_norm)

        # Point-wise Convolution (Query, Key, Value)
        Q_i = self.q_conv1(F_i_dw)
        K_i = self.k_conv1(F_i_dw)
        V_i = self.v_conv1(F_i_dw)

        Q_w = self.q_conv2(F_w_dw)
        K_w = self.k_conv2(F_w_dw)
        V_w = self.v_conv2(F_w_dw)

        # Depth-wise Convolution after Point-wise Convolution
        Q_i = self.q_dw_conv1(Q_i).view(B, H * W, -1)
        K_i = self.k_dw_conv1(K_i).view(B, H * W, -1)
        V_i = self.v_dw_conv1(V_i).view(B, -1, H * W)

        Q_w = self.q_dw_conv2(Q_w).view(B, H * W, -1)
        K_w = self.k_dw_conv2(K_w).view(B, H * W, -1)
        V_w = self.v_dw_conv2(V_w).view(B, -1, H * W)

        # Attention Mechanism
        attention_1 = F.softmax(torch.bmm(Q_i.transpose(1, 2), K_w), dim=-1)
        attention_2 = F.softmax(torch.bmm(Q_w.transpose(1, 2), K_i), dim=-1)

        A_w = torch.bmm(attention_1, V_w).view(B, H, W, C2)
        A_i = torch.bmm(attention_2, V_i).view(B, H, W, C1)

        A_w = A_w.permute(0, 3, 1, 2)
        A_i = A_i.permute(0, 3, 1, 2)

        # Concatenate and Linear Projection
        concat = torch.cat((A_w, A_i), dim=1)
        B, C, H, W = concat.shape
        concat = concat.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)  # 展平特征图
        output = self.linear_proj(concat)
        output = output.view(B, -1, H, W).permute(0, 2, 3, 1).contiguous().view(B, -1, H, W)  # 恢复特征图的形状
        # output = self.linear_proj(concat)

        # Residual Connection with channel matching
        # F_i_res = self.residual_conv1(F_i)
        # F_w_res = self.residual_conv2(F_w)

        output = output + F_i + F_w
        # output = torch.cat((output, F_i, F_w), dim=1)

        return output


from torch.nn import Softmax


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class DepthwiseSeparableReducer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableReducer, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.leaky_relu(x)
        return x


class double_CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(double_CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.depth = DepthwiseSeparableReducer(in_dim * 2, in_dim)

    def forward(self, x, y):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        ##############################################
        proj_query2 = self.query_conv(y)
        proj_query_H2 = proj_query2.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                   1)
        proj_query_W2 = proj_query2.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                   1)
        proj_key2 = self.key_conv(y)
        proj_key_H2 = proj_key2.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W2 = proj_key2.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value2 = self.value_conv(y)
        proj_value_H2 = proj_value2.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W2 = proj_value2.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        ######################################################
        inf_ = self.INF(m_batchsize, height, width).to("cuda")
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + inf_).view(m_batchsize, width, height, height).permute(0, 2,
                                                                                                                 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H2, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3,
                                                                                                              1)
        out_W = torch.bmm(proj_value_W2, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1,
                                                                                                              3)
        ##############################################
        energy_H2 = (torch.bmm(proj_query_H2, proj_key_H2) + inf_).view(m_batchsize, width, height, height).permute(0,
                                                                                                                    2,
                                                                                                                    1,
                                                                                                                    3)
        energy_W2 = torch.bmm(proj_query_W2, proj_key_W2).view(m_batchsize, height, width, width)
        concate2 = self.softmax(torch.cat([energy_H2, energy_W2], 3))

        att_H2 = concate2[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W2 = concate2[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H2 = torch.bmm(proj_value_H, att_H2.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3,
                                                                                                               1)
        out_W2 = torch.bmm(proj_value_W, att_W2.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1,
                                                                                                               3)
        # out1 = out_H + out_W
        # out2 = out_H2 + out_W2
        # out = torch.cat([out1, out2], dim=1)
        # out = self.depth(out)
        # return self.gamma * out + x + y

        out1 = self.gamma1*(out_H + out_W) + x
        out2 = self.gamma2*(out_H2 + out_W2) + y
        out = torch.cat([out1, out2], dim=1)
        out = self.depth(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // reduction, 1, bias=False),
            NonLinearity(),
            nn.Conv2d(in_dim // reduction, in_dim, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class EnhancedCrossAttentionFusion(nn.Module):
    def __init__(self, in_dim_x, in_dim_y, out_dim):
        super(EnhancedCrossAttentionFusion, self).__init__()
        self.conv_x = nn.Conv2d(in_dim_x, out_dim, kernel_size=3, padding=1)
        self.conv_y = nn.Conv2d(in_dim_y, out_dim, kernel_size=3, padding=1)
        self.attention_conv = nn.Conv2d(out_dim * 2, 1, kernel_size=1)

        self.channel_attention_x = ChannelAttention(out_dim)
        self.channel_attention_y = ChannelAttention(out_dim)
        self.spatial_attention = SpatialAttention()

        self.fusion_conv = nn.Conv2d(out_dim * 2, out_dim, kernel_size=1)

    def forward(self, x, y):
        conv_x = self.conv_x(x)
        conv_y = self.conv_y(y)

        concat_features = torch.cat([conv_x, conv_y], dim=1)
        attention_weights = torch.sigmoid(self.attention_conv(concat_features))

        ca_x = self.channel_attention_x(conv_x) * conv_x
        ca_y = self.channel_attention_y(conv_y) * conv_y

        fused_features = attention_weights * ca_x + (1 - attention_weights) * ca_y

        sa = self.spatial_attention(fused_features)
        fused_features = sa * fused_features

        output = self.fusion_conv(torch.cat([fused_features, x + y], dim=1))

        return output

