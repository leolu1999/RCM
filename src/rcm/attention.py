import torch
from torch import nn
from typing import List, Callable
import warnings
import torch.nn.functional as F
from src.rcm.utils.utils import TransLN_2d
try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, 'scaled_dot_product_attention'):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True



class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                'FlashAttention is not available. For optimal speed, '
                'consider installing torch >= 2.0 or flash-attn.',
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        if allow_flash and FlashCrossAttention:
            self.flash_ = FlashCrossAttention()

    def forward(self, q, k, v) -> torch.Tensor:
        if self.enable_flash and q.device.type == 'cuda':
            if FlashCrossAttention:
                q, k, v = [x.transpose(-2, -3) for x in [q, k, v]]
                m = self.flash_(q.half(), torch.stack([k, v], 2).half())
                return m.transpose(-2, -3).to(q.dtype)
            else:  # use torch 2.0 scaled_dot_product_attention with flash
                args = [x.half().contiguous() for x in [q, k, v]]
                with torch.backends.cuda.sdp_kernel(enable_flash=True):
                    return F.scaled_dot_product_attention(*args).to(q.dtype)
        elif hasattr(F, 'scaled_dot_product_attention'):
            args = [x.contiguous() for x in [q, k, v]]
            return F.scaled_dot_product_attention(*args).to(q.dtype)
        else:
            s = q.shape[-1] ** -0.5
            attn = F.softmax(torch.einsum('...id,...jd->...ij', q, k) * s, -1)
            return torch.einsum('...ij,...jd->...id', attn, v)


class SelfAttn(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim)
        )

    def _forward(self, x: torch.Tensor,
                 mask=None):
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        if mask is not None:
            mask = mask[:, None, :, None]
            k.masked_fill_(~mask, 0.)
            v.masked_fill_(~mask, 0.)
        context = self.inner_attn(q, k, v)
        message = self.out_proj(
            context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))

    def forward(self, x0, x1, mask0=None, mask1=None):
        return self._forward(x0, mask0), self._forward(x1, mask1)


class CrossAttn(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim)
        )

        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, mask0=None, mask1=None) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1))
        if mask0 is not None:
            mask = mask0[:, None, :, None]
            qk0.masked_fill_(~mask, 0.)
            v0.masked_fill_(~mask, 0.)
        if mask1 is not None:
            mask = mask1[:, None, :, None]
            qk1.masked_fill_(~mask, 0.)
            v1.masked_fill_(~mask, 0.)
        if self.flash is not None:
            m0 = self.flash(qk0, qk1, v1)
            m1 = self.flash(qk1, qk0, v0)
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum('b h i d, b h j d -> b h i j', qk0, qk1)
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum('bhij, bhjd -> bhid', attn01, v1)
            m1 = torch.einsum('bhji, bhjd -> bhid', attn10.transpose(-2, -1), v0)
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2),
                           m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class CrossAttn_d2s(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim)
        )

        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v1 = self.to_v(x1)
        qk0, qk1, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v1))
        if self.flash is not None:
            m0 = self.flash(qk0, qk1, v1)
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum('b h i d, b h j d -> b h i j', qk0, qk1)
            attn01 = F.softmax(sim, dim=-1)
            m0 = torch.einsum('bhij, bhjd -> bhid', attn01, v1)
        m0 = m0.transpose(1, 2).flatten(start_dim=-2)
        m0 = self.to_out(m0)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        return x0


class CrossLin_s2d(nn.Module):
    def __init__(self, feature_dim: int, window=5):
        super().__init__()
        self.window = window
        self.lin = nn.Conv1d(feature_dim, feature_dim, 1)
        self.mlp_dense = nn.Sequential(
            nn.Conv2d(2*feature_dim, feature_dim, kernel_size=1),
            TransLN_2d(feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x, source):
        m, ww, c = x.shape
        x, source = x.transpose(1, 2), source.transpose(1, 2)
        message = self.lin(source).repeat(1, 1, ww)
        message = torch.cat([x, message], dim=1).view(m, 2*c, self.window, -1)
        output = self.mlp_dense(message).view(m, c, -1)
        output = x + output
        return output.transpose(1, 2)
