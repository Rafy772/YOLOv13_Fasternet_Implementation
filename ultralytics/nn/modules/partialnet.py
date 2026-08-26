# ======================================================================
# partialnet.py -- PartialNet modules for YOLOv13 (iMoonLab fork)
# v1.3
#   - MLP activation GELU -> ReLU(inplace) to match the official config
#     (all released PartialNet variants use act_layer='RELU')
#   - use_sp now defaults to True (official use_spatial_attn=True)
#   - head picker prefers the official 6/4-head settings
#   - RPBAttention no longer pickles its coordinate cache
#
# Implements the Partial Channel Mechanism from:
#   Huang et al., "Partial Channel Network: Compute Fewer, Perform
#   Better", AAAI 2026. arXiv:2502.01303
#   Official code: https://github.com/haiduo/PartialNet
#
# Modules
#   SRM             -- Gaussian channel attention (mean+std style pooling),
#                      the "enhanced Gaussian-SE" used inside PAT_ch.
#   PATConv         -- Partial Attention Convolution. Splits channels:
#                      dim/n_div -> Conv3x3, rest -> visual attention,
#                      concat. attn='se' (PAT_ch) or attn='sf' (PAT_sf).
#   PATSpatialAttn  -- PAT_sp. 50/50 split: one half -> 1-channel
#                      Hardsigmoid spatial map, other half -> Conv1x1.
#                      Placed after the MLP (official "reverse" layout).
#   RPBAttention    -- self-attention for PAT_sf. NOTE: the official repo
#                      uses iRPE (euclidean bucket RPE). To stay
#                      self-contained and resolution-agnostic, this file
#                      substitutes a Swin-v2-style continuous relative
#                      position bias (log-CPB MLP). Document this
#                      substitution when writing the paper.
#   PartialNetBlock -- PAT_ch/PAT_sf + MLP(1x1->BN->GELU->1x1[->PAT_sp])
#                      with residual, mlp_ratio=2 (official T0..L config).
#   C3k2_PAT        -- C2f-style wrapper: drop-in replacement for
#                      C3k2/DSC3k2 with PartialNetBlock inner modules.
#
# Deliberately NOT registered as a standalone YAML module: the raw
# partial conv (same lesson as the PConv positional-arg parsing bug --
# only block-level modules with (c1, c2, ...) signatures are exposed).
# ======================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .block import C2f
from .conv import Conv

__all__ = ("SRM", "PATConv", "PATSpatialAttn", "RPBAttention",
           "PartialNetBlock", "C3k2_PAT", "PATStage")


class _BN2dSafe(nn.BatchNorm2d):
    """BatchNorm2d that falls back to running statistics when a train-mode
    input has only one value per channel (batch=1 on a 1x1 map), instead of
    raising. Identical to BatchNorm2d whenever batch > 1.

    Needed because SRM's BN sees (b, c, 1, 1) tensors: Ultralytics profiles
    GFLOPs with a single train-mode image, and a real run would crash if an
    epoch's last batch contains one image. Running stats are still updated
    on every normal (batch > 1) step, and the affine transform (and its
    gradients) applies in the fallback too.
    """

    def forward(self, x):
        if self.training and x.shape[0] * x.shape[2] * x.shape[3] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


class SRM(nn.Module):
    """Gaussian channel attention ("style pooling": per-channel mean+std).

    Faithful to the official implementation: cat([mean, std]) -> full
    Conv2d(c, c, kernel=(1,2)) (cross-channel mixing, the "enhanced"
    part vs. plain SRM/SE) -> BN -> Hardsigmoid -> channel scaling.
    """

    def __init__(self, channel):
        super().__init__()
        self.cfc1 = nn.Conv2d(channel, channel, kernel_size=(1, 2), bias=False)
        self.bn = _BN2dSafe(channel)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        mean = x.reshape(b, c, -1).mean(-1).view(b, c, 1, 1)
        std = x.reshape(b, c, -1).std(-1).view(b, c, 1, 1)
        u = torch.cat([mean, std], dim=-1)          # (b, c, 1, 2)
        z = self.bn(self.cfc1(u))                   # (b, c, 1, 1)
        g = self.sigmoid(z).reshape(b, c, 1, 1)
        return x * g.expand_as(x)

    def fuse_bn(self):
        """Fold BN into cfc1 for inference (idempotent)."""
        if isinstance(self.bn, nn.BatchNorm2d):
            self.cfc1 = fuse_conv_and_bn(self.cfc1, self.bn)
            self.bn = nn.Identity()


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors (timm-style)."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class RPBAttention(nn.Module):
    """Multi-head self-attention with continuous relative position bias.

    Stand-in for the official RPEAttention (iRPE, method='euc', rpe on
    keys). Uses a small MLP on log-spaced relative coordinates
    (Swin-v2 log-CPB), which works at any feature-map resolution --
    important for YOLO where train/val/deploy imgsz may differ.
    Dropout defaults (0.1/0.1) match the official PAT_sf settings.
    """

    def __init__(self, dim, num_heads, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by heads {num_heads}"
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_heads, bias=False),
        )
        self._table_cache = {}   # (h, w) -> (M, 2) log-spaced displacement coords
        self._index_cache = {}   # (h, w) -> (N*N,) long index into the table

    def __getstate__(self):
        """Exclude the caches from pickles (torch.save of whole models
        would otherwise embed per-resolution tables in every checkpoint)."""
        state = self.__dict__.copy()
        state["_table_cache"] = {}
        state["_index_cache"] = {}
        return state

    def _rel_bias(self, h, w, device):
        """Relative position bias (heads, N, N), Swin-v2 style.

        The CPB MLP runs on the (2h-1)*(2w-1) displacement table (M
        entries) and is gathered per token pair -- evaluating it per
        pair directly would cost ~N^2/M ~ 100x more at YOLO map sizes.
        """
        key = (h, w)
        tab = self._table_cache.get(key)
        idx = self._index_cache.get(key)
        if tab is None or tab.device != device:
            dy = torch.arange(-(h - 1), h, device=device, dtype=torch.float32)
            dx = torch.arange(-(w - 1), w, device=device, dtype=torch.float32)
            gy, gx = torch.meshgrid(dy, dx, indexing="ij")
            tab = torch.stack([gy, gx], dim=-1).reshape(-1, 2)      # (M, 2)
            tab = torch.sign(tab) * torch.log1p(tab.abs()) / torch.log(
                torch.tensor(8.0, device=device))
            ys, xs = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device),
                indexing="ij")
            cy, cx = ys.flatten(), xs.flatten()                     # (N,)
            ry = cy[:, None] - cy[None, :] + (h - 1)                # (N, N)
            rx = cx[:, None] - cx[None, :] + (w - 1)
            idx = (ry * (2 * w - 1) + rx).reshape(-1)               # (N*N,)
            self._table_cache[key] = tab
            self._index_cache[key] = idx
        n = h * w
        table = self.cpb_mlp(tab)                                   # (M, heads)
        return table[idx].reshape(n, n, -1).permute(2, 0, 1)        # (heads, N, N)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        t = x.flatten(2).transpose(1, 2)                                  # (b, N, c)
        qkv = self.qkv(t).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                              # (b, heads, N, hd)
        attn = (q * self.scale) @ k.transpose(-2, -1)                     # (b, heads, N, N)
        bias = self._rel_bias(h, w, x.device)                             # (heads, N, N)
        attn = attn + bias.unsqueeze(0).to(attn.dtype)
        attn = self.attn_drop(attn.softmax(dim=-1))
        t = (attn @ v).transpose(1, 2).reshape(b, n, c)
        t = self.proj_drop(self.proj(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


def _pick_heads(c):
    """Head count dividing c with head_dim >= 16, preferring the official
    PartialNet choices (6 heads for T1..L, 4 for T0), then fallbacks."""
    for hd in (6, 4, 8, 3, 2, 1):
        if c % hd == 0 and c // hd >= 16:
            return hd
    return 1


class PATConv(nn.Module):
    """Partial Attention Convolution (Eq. 1 of the paper).

    Channels split into [dim/n_div | rest]; the first part goes through a
    dense Conv3x3, the rest through visual attention, run in parallel and
    concatenated.

    attn='se' -> PAT_ch: SRM attention, BN applied AFTER attention.
    attn='sf' -> PAT_sf: LayerNorm2d applied BEFORE RPB self-attention.
    Branch/normalization order copies the official Partial_conv3.
    """

    def __init__(self, dim, n_div=4, attn="se"):
        super().__init__()
        self.dim_conv = max(dim // n_div, 1)
        self.dim_untouched = dim - self.dim_conv
        self.attn_type = attn
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
        if attn == "se":
            self.attn = SRM(self.dim_untouched)
            self.norm = nn.BatchNorm2d(self.dim_untouched)
        elif attn == "sf":
            self.attn = RPBAttention(self.dim_untouched, _pick_heads(self.dim_untouched))
            self.norm = LayerNorm2d(self.dim_untouched)
        else:
            raise ValueError(f"PATConv attn must be 'se' or 'sf', got {attn!r}")

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        if self.attn_type == "se":
            x2 = self.norm(self.attn(x2))     # attn -> BN (official order)
        else:
            x2 = self.attn(self.norm(x2))     # LN -> attn (official order)
        return torch.cat((x1, x2), 1)


class PATSpatialAttn(nn.Module):
    """PAT_sp ("reverse" layout, official partial_spatial_attn_layer_reverse).

    50/50 channel split: first half -> pointwise conv to n_head(=1) map ->
    Hardsigmoid -> spatial reweighting -> BN; second half -> BN -> Conv1x1
    (mergeable with the preceding MLP Conv1x1 at deploy time); concat.
    """

    def __init__(self, dim, n_head=1, partial=0.5):
        super().__init__()
        self.dim_conv = int(partial * dim)
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 1, bias=False)
        self.conv_attn = nn.Conv2d(self.dim_untouched, n_head, 1, bias=False)
        self.norm = nn.BatchNorm2d(self.dim_untouched)
        self.norm2 = nn.BatchNorm2d(self.dim_conv)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_untouched, self.dim_conv], 1)
        x1 = self.norm(x1 * self.act(self.conv_attn(x1)))
        x2 = self.conv(self.norm2(x2))
        return torch.cat((x1, x2), 1)


class PartialNetBlock(nn.Module):
    """One PartialNet block: PATConv + ConvFFN (+ optional PAT_sp).

    attn='se' (official 'se' path):   out = x + MLP(PATConv(x))
    attn='sf' (official 'self' path): x = x + PATConv(x); out = x + MLP(x)
    MLP = Conv1x1(dim->dim*mlp_ratio, no bias) -> BN -> ReLU ->
          Conv1x1(->dim, no bias) [-> PAT_sp]. mlp_ratio=2 and ReLU per the
    official config (every released PartialNet variant, incl. the COCO
    detection backbones, uses act_layer='RELU').
    use_sp defaults to True: the official models apply PAT_sp in every
    block (use_spatial_attn=True); pass False to ablate it.
    """

    def __init__(self, dim, n_div=4, mlp_ratio=2.0, use_sp=True, attn="se"):
        super().__init__()
        self.attn_type = attn
        self.mix = PATConv(dim, n_div, attn)
        hidden = int(dim * mlp_ratio)
        layers = [
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
        ]
        if use_sp:
            layers.append(PATSpatialAttn(dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.attn_type == "sf":
            x = x + self.mix(x)
            return x + self.mlp(x)
        return x + self.mlp(self.mix(x))

    def fuse_bn(self):
        """Fold the fusable BNs for inference (idempotent).

        Ultralytics fuse() only folds its own Conv wrappers, so plain
        nn.Conv2d+BN pairs in custom blocks stay unfused unless handled
        here (same issue previously hit with FasterNetBlock). Fuses the
        MLP's Conv1x1+BN and SRM's cfc1+BN. The post-attention BN in
        PAT_ch and the BNs in PAT_sp are not conv-adjacent and stay.
        """
        if len(self.mlp) >= 2 and isinstance(self.mlp[1], nn.BatchNorm2d):
            fused = fuse_conv_and_bn(self.mlp[0], self.mlp[1])
            self.mlp = nn.Sequential(fused, *list(self.mlp)[2:])
        if isinstance(getattr(self.mix, "attn", None), SRM):
            self.mix.attn.fuse_bn()


class C3k2_PAT(C2f):
    """C3k2/DSC3k2 drop-in with PartialNetBlock inner modules.

    YAML args after parse_model: (c1, c2, n, use_sp, e, n_div, attn, mlp_ratio)
      - [-1, 2, C3k2_PAT, [256, False, 0.25]]          # PAT_ch, e=0.25
      - [-1, 2, C3k2_PAT, [512, True]]                 # PAT_ch + PAT_sp
      - [-1, 4, C3k2_PAT, [1024, True, 0.5, 4, "sf"]]  # PAT_sf (last stage)
    """

    def __init__(self, c1, c2, n=1, use_sp=False, e=0.5, n_div=4, attn="se",
                 mlp_ratio=2.0):
        super().__init__(c1, c2, n, False, 1, e)
        self.m = nn.ModuleList(
            PartialNetBlock(self.c, n_div, mlp_ratio, use_sp, attn)
            for _ in range(n)
        )


class PATStage(nn.Module):
    """Plain PartialNet stage (paper-style BasicStage): optional 1x1
    projection to the stage width, then n stacked PartialNetBlocks at
    full width. No CSP split -- matches the sequential MetaFormer layout
    of the official backbone.

    YAML args after parse_model: (c1, c2, n, use_sp, n_div, attn, mlp_ratio)
      - [-1, 2, PATStage, [256]]                # PAT_ch stage
      - [-1, 4, PATStage, [1024, False, 4, "sf"]]  # PAT_sf stage (last)
    """

    def __init__(self, c1, c2, n=1, use_sp=False, n_div=4, attn="se",
                 mlp_ratio=2.0):
        super().__init__()
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(
            PartialNetBlock(c2, n_div, mlp_ratio, use_sp, attn)
            for _ in range(n)
        ))

    def forward(self, x):
        return self.blocks(self.proj(x))
