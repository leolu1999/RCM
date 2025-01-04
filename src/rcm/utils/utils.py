import torch
from torch import nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
INF = 1E4


class Assign(nn.Module):
    def __init__(self, dim, learnable_temp):
        super().__init__()
        self.assign_attn = assign(dim, learnable_temp)

    def forward(self, sparse, dense, mask0, mask1, mode='test'):
        if mode == 'test':
            attn_dict = self.assign_attn(sparse, dense, mask0, mask1)
        else:
            attn_dict = self.assign_attn.train_forward(sparse, dense, mask0, mask1)
        return attn_dict


class assign(nn.Module):
    def __init__(self, dim, learnable_temp):
        super().__init__()
        self.dim = dim
        self.scale = self.dim**-0.5
        self.proj = nn.Conv1d(dim, dim, kernel_size=1)
        if learnable_temp:
            self.temp = torch.nn.Parameter(torch.tensor(0.1), requires_grad=True)
        else:
            self.temp = torch.tensor(1.)

    def forward(self, sparse, dense, mask0=None, mask1=None):
        dense, sparse = self.proj(dense), self.proj(sparse)
        raw_attn = self.scale * torch.einsum('bdn,bdm->bnm', sparse, dense) / self.temp
        if mask0 is not None:
            raw_attn[:, :, :].masked_fill_(~mask0[..., None], -INF)
        if mask1 is not None:
            raw_attn[:, :, :].masked_fill_(~mask1[:, None, :], -INF)
        soft_attn = F.softmax(raw_attn, dim=2)
        return soft_attn

    def train_forward(self, sparse, dense, mask0=None, mask1=None):
        b, c, _, l = dense.shape
        dense, sparse = self.proj(dense.view(b, c, -1)).view(b, c, -1, l), self.proj(sparse.view(b, c, -1)).view(b, c, -1, l)
        raw_attn = self.scale * torch.einsum('bdnl,bdml->bnml', sparse, dense) / self.temp
        if mask0 is not None:
            raw_attn[:, :, :, :].masked_fill_(~mask0[..., None, None], -INF)
        if mask1 is not None:
            raw_attn[:, :, :, :].masked_fill_(~mask1[:, None, :, None], -INF)
        soft_attn = F.softmax(raw_attn, dim=2)
        return soft_attn


class TransLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x.transpose(1,2)).transpose(1,2)


class TransLN_2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        _, _, h, _ = x.shape
        x = rearrange(x, 'b d h w->b (h w) d')
        x = self.ln(x)
        return rearrange(x, 'b (h w) d->b d h w', h=h)


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(TransLN(channels[i]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def MLP_2d(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv2d(channels[i - 1], channels[i], kernel_size=3, padding=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(TransLN(channels[i]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
        kpts: torch.Tensor,
        size: torch.Tensor) -> torch.Tensor:
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts, scores]
        return self.encoder(torch.cat(inputs, dim=1))


def switch_back4(data):
    if 'mask1' in data:
        keys0 = ['image0', 'depth0', 'K0', 'scale0', 'T_0to1', 'prepad_size0']
        keys1 = ['image1', 'depth1', 'K1', 'scale1', 'T_1to0', 'prepad_size1']
    else:
        keys0 = ['image0', 'depth0', 'K0', 'T_0to1']
        keys1 = ['image1', 'depth1', 'K1', 'T_1to0']
    for b in range(len(data['image0'])):
        if data['switch'][b]:
            b_mask = data['m_bids'] == b
            temp = data['mkpts0_f'][b_mask]
            data['mkpts0_f'][b_mask] = data['mkpts1_f'][b_mask]
            data['mkpts1_f'][b_mask] = temp
    # switch
    for (key0, key1) in zip(keys0, keys1):
        item0_list, item1_list = [], []
        for b in range(len(data[key0])):
            if data['switch'][b]:
                item0_list.append(data[key1][b])
                item1_list.append(data[key0][b])
            else:
                item0_list.append(data[key0][b])
                item1_list.append(data[key1][b])
        data.update({
            key0: torch.stack(item0_list, 0),
            key1: torch.stack(item1_list, 0)
        })
