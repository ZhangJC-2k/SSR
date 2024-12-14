import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
import time


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type='ln'):
        super().__init__()
        self.fn = fn
        self.norm_type = norm_type
        if norm_type == 'ln':
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.GroupNorm(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, *args, **kwargs):
        if self.norm_type == 'ln':
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class Para_Estimator(nn.Module):
    def __init__(self, in_nc=28, out_nc=1, channel=32):
        super(Para_Estimator, self).__init__()
        self.fusion = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.bias = nn.Parameter(torch.FloatTensor([1.]))
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=False),
                )
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):

        x = self.relu(self.fusion(x))
        x = self.avpool(x)
        x = self.mlp(x) + self.bias
        return x


class WSSA(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1, shift=False):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift = shift

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)
        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def cal_attention(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        h1, h2 = h // self.window_size[0], w // self.window_size[1]
        q, k, v = map(lambda t: rearrange(t, 'b c (h1 h) (h2 w) ->b (h1 h2) c (h w)', h1=h1, h2=h2), (q, k, v))
        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b (h1 h2) c (h w) -> b c (h1 h) (h2 w)', h1=h1, h=h // h1)
        out = self.to_out(out)
        return out

    def forward(self, x):

        w_size = self.window_size
        if self.shift:
            x = x.roll(shifts=w_size[0]//2, dims=2).roll(shifts=w_size[1]//2, dims=3)
        out = self.cal_attention(x)
        if self.shift:
            out = out.roll(shifts=-1*w_size[1]//2, dims=3).roll(shifts=-1*w_size[0]//2, dims=2)
        return out


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):

        out = self.net(x)
        return out


class ERB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1):
        super().__init__()
        self.WSSA = PreNorm(dim, WSSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads,
                                      shift=False))
        self.FFN = PreNorm(dim, FFN(dim=dim), norm_type='gn')

    def forward(self, x):

        x = self.WSSA(x) + x
        x = self.FFN(x) + x
        return x


class CMB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.to_a = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 11, 1, 5, groups=dim, bias=False),
        )
        self.to_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def cal_attention(self, x):
        a, v = self.to_a(x), self.to_v(x)
        out = self.to_out(a*v)
        return out

    def forward(self, x):
        out = self.cal_attention(x)
        return out


class SAB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.Estimator = nn.Sequential(
            nn.Conv2d(dim, 1, 3, 1, 1, bias=False),
            GELU(),
        )
        self.SW = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight.data, mean=0.0, std=.02)

    def forward(self, f):
        f = self.conv(f)
        out = self.SW(f) * self.Estimator(f).repeat(1, self.dim, 1, 1)
        out = self.out(out)
        return out


class ARB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.CMB = PreNorm(dim, CMB(dim=dim))
        self.SAB = PreNorm(dim, SAB(dim=dim), norm_type='gn')

    def forward(self, x):

        x = self.CMB(x) + x
        x = self.SAB(x) + x
        return x


class SSRB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1):
        super().__init__()

        self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.SARB = ERB(dim, window_size, dim_head, heads)
        self.SRB = ARB(dim)

    def forward(self, x):

        x = self.pos(x) + x
        x = self.SARB(x)
        x = self.SRB(x)

        return x


class Mask_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv2d(28, 28, 3, 1, 1, bias=False)
        self.mask = nn.Conv2d(28, 28, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        out = self.cnn(x)*(1+self.mask(mask))
        return out


class SSRU(nn.Module):
    def __init__(self, in_dim=56, out_dim=28):
        super(SSRU, self).__init__()

        self.mask_embedding = Mask_embedding()
        self.down1 = SSRB(dim=28, dim_head=28, heads=1)
        self.downsample1 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)
        self.down2 = SSRB(dim=56, dim_head=28, heads=2)
        self.downsample2 = nn.Conv2d(56, 112, 4, 2, 1, bias=False)
        self.bottleneck = SSRB(dim=112, dim_head=28, heads=4)
        self.upsample2 = nn.ConvTranspose2d(112, 56, 2, 2)
        self.fusion2 = nn.Conv2d(112, 56, 1, 1, 0, bias=False)
        self.up2 = SSRB(dim=56, dim_head=28, heads=2)
        self.upsample1 = nn.ConvTranspose2d(56, 28, 2, 2)
        self.fusion1 = nn.Conv2d(56, 28, 1, 1, 0, bias=False)
        self.up1 = SSRB(dim=28, dim_head=28, heads=1)
        self.out = nn.Conv2d(28, out_dim, 3, 1, 1, bias=False)

    def forward(self, x, mask):

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x_in = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        mask = F.pad(mask, [0, pad_w, 0, pad_h], mode='reflect')

        x = self.mask_embedding(x_in, mask)
        x1 = self.down1(x)
        x = self.downsample1(x1)
        x2 = self.down2(x)
        x = self.downsample2(x2)
        x = self.bottleneck(x)
        x = self.upsample2(x)
        x = self.fusion2(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.upsample1(x)
        x = self.fusion1(torch.cat([x, x1], dim=1))
        x = self.up1(x)
        out = self.out(x) + x_in

        return out[:, :, :h_inp, :w_inp]


class Net(torch.nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        netlayer = []
        self.stage = opt.stage
        self.nC = opt.bands
        self.size = opt.size
        self.initial = nn.Conv2d(self.nC * 2, self.nC, 1, 1, 0)
        para_estimator = []
        for i in range(opt.stage):
            para_estimator.append(Para_Estimator())

        for i in range(opt.stage):
            netlayer.append(SSRU(in_dim=56))
            netlayer.append(ARB(28))

        self.rhos = nn.ModuleList(para_estimator)
        self.net_stage = nn.ModuleList(netlayer)

    def shift_back(self, x, len_shift=2):
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(-1) * len_shift * i, dims=2)
        return x[:, :, :, :self.size]

    def shift(self, x, len_shift=2):
        x = F.pad(x, [0, self.nC*2-2, 0, 0], mode='constant', value=0)
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=len_shift * i, dims=2)
        return x

    def mul_PhiTg(self, Phi_shift, g):
        temp_1 = g.repeat(1, Phi_shift.shape[1], 1, 1).cuda()
        PhiTg = temp_1 * Phi_shift
        PhiTg = self.shift_back(PhiTg)
        return PhiTg

    def mul_Phif(self, Phi_shift, z):
        z_shift = self.shift(z)
        Phiz = Phi_shift * z_shift
        Phiz = torch.sum(Phiz, 1)
        return Phiz.unsqueeze(1)

    def forward(self, g, input_mask=None):
        Phi, PhiPhiT = input_mask
        Phi_shift = self.shift(Phi, len_shift=2)
        g_normal = g / self.nC*2
        temp_g = g_normal.repeat(1, 28, 1, 1)
        f0 = self.shift_back(temp_g)
        f = self.initial(torch.cat([f0, Phi], dim=1))

        out = []
        for i in range(self.stage):

            '''LMP'''
            rho = self.rhos[i](f)
            Phi_f = self.mul_Phif(Phi_shift, f)
            z = f + rho * self.mul_PhiTg(Phi_shift, torch.div(g - Phi_f, PhiPhiT))
            '''SSRU'''
            f = self.net_stage[2 * i](z, Phi)
            '''ARB'''
            f = self.net_stage[2 * i + 1](f)
            out.append(f)

        return out


