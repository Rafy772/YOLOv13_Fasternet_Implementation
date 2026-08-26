"""FasterNet modules for YOLOv13.

Paper: "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks"
       Jierun Chen et al., CVPR 2023.  https://arxiv.org/abs/2303.03667

Core idea of the paper: many "efficient" operators (depthwise conv, etc.) reduce
FLOPs but also reduce FLOPS (throughput) because of frequent memory access.
Partial Convolution (PConv) applies a regular k x k convolution to only a
fraction (1 / n_div, typically 1/4) of the input channels and leaves the rest
untouched, cutting both FLOPs (~1/16 of a full conv at r=1/4) and memory
access (~1/4). A pointwise-conv MLP afterwards mixes information across all
channels, giving an effective T-shaped receptive field (Sec. 3.2-3.3).

Integration notes for YOLOv13 (Ultralytics fork):
* PConv is deliberately NOT registered as a standalone YAML module.
  parse_model rewrites args as (c1, c2, ...), which corrupts PConv's
  (dim, n_div, ...) signature. PConv is only instantiated inside
  FasterNetBlock.
* C3k2_Faster is a drop-in replacement for DSC3k2 / C3k2 in the YAML:
  the positional args [c2, c3k, e] map identically.
* The FasterNet block keeps the paper layout exactly: PConv -> PWConv
  (expand by MLP_RATIO) -> BN -> act -> PWConv (project), with a residual
  shortcut. Normalization/activation appear ONLY after the middle layer
  (Sec. 3.4: "to preserve feature diversity and achieve lower latency").
"""

import copy

import torch
import torch.nn as nn

from .block import C2f, C3k
from .conv import Conv
from .head import Detect

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PCONV_N_DIV = 4            # paper: conv applied to 1/4 of channels (r = 1/4)
PCONV_KERNEL = 3           # paper: 3x3 partial conv (set 5 for the Kernel-5 ablation)
MLP_RATIO = 2.0            # paper: PWConv hidden expansion ratio of 2
FASTERNET_ACT = nn.ReLU    # paper: ReLU for larger variants, GELU for T0/T1
DROP_PATH = 0.0            # stochastic depth rate (paper uses 0.0-0.1 on ImageNet)
LAYER_SCALE_INIT = 0.0     # 0 disables LayerScale (official default: disabled)
PCONV_FORWARD = "split_cat"  # "split_cat" (train + infer) | "slicing" (infer only)
# ---------------------------------------------------------------------------

__all__ = ("PConv", "FasterNetBlock", "C3k_Faster", "C3k2_Faster", "TConv", "DetectFaster")


class DropPath(nn.Module):
    """Per-sample stochastic depth (embedded here to avoid a timm dependency)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        return x * mask / keep_prob


class PConv(nn.Module):
    """Partial Convolution (PConv), Sec. 3.2 / Fig. 4 of the FasterNet paper.

    Applies a regular k x k conv to the first `dim // n_div` channels and the
    identity to the remaining channels. bias=False and no norm/act, exactly as
    in the official implementation (norm/act live in the following MLP).

    NOTE: do not register this class in parse_model / the YAML module tables.
    Its signature is (dim, n_div, ...), not (c1, c2, ...): YAML arg rewriting
    would silently pass the output-channel count as n_div.
    """

    def __init__(self, dim, n_div=PCONV_N_DIV, k=PCONV_KERNEL, forward_type=PCONV_FORWARD):
        super().__init__()
        assert dim // n_div > 0, f"PConv needs dim ({dim}) >= n_div ({n_div})"
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, k, 1, k // 2, bias=False)
        if forward_type not in ("split_cat", "slicing"):
            raise NotImplementedError(f"PConv forward_type: {forward_type}")
        self.forward_type = forward_type

    def forward(self, x):
        if self.forward_type == "slicing":  # inference only (in-place on a clone)
            x = x.clone()
            x[:, : self.dim_conv] = self.conv(x[:, : self.dim_conv])
            return x
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        return torch.cat((self.conv(x1), x2), 1)


class FasterNetBlock(nn.Module):
    """FasterNet block (Fig. 4 of the paper).

    x -> PConv -> PWConv(c -> c*MLP_RATIO) -> BN -> act -> PWConv(-> c) -> (+x)

    The residual shortcut is intrinsic to the block (input/output dims match),
    so the YAML-level `shortcut` flag of the surrounding C3k2 is irrelevant
    here. If c1 != c2 a 1x1 Conv adjusts channels before the block.
    """

    def __init__(
        self,
        c1,
        c2=None,
        n_div=PCONV_N_DIV,
        mlp_ratio=MLP_RATIO,
        k=PCONV_KERNEL,
        drop_path=DROP_PATH,
        layer_scale_init=LAYER_SCALE_INIT,
        act=FASTERNET_ACT,
    ):
        super().__init__()
        c2 = c2 or c1
        self.adjust = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        hidden = int(c2 * mlp_ratio)
        self.pconv = PConv(c2, n_div, k)
        self.mlp = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            act(),
            nn.Conv2d(hidden, c2, 1, bias=False),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale = (
            nn.Parameter(layer_scale_init * torch.ones(c2), requires_grad=True)
            if layer_scale_init > 0.0
            else None
        )

    def fuse_bn(self):
        """Fold the MLP's BatchNorm into the preceding 1x1 conv (inference).

        Ultralytics' BaseModel.fuse() only folds BN inside its own Conv/DWConv
        wrappers, so without this the block carries an unfused BN per forward.
        Safe to call repeatedly; trained checkpoints load unchanged (folding
        happens in memory at fuse time, not at init).
        """
        if isinstance(self.mlp[1], nn.BatchNorm2d):
            from ultralytics.utils.torch_utils import fuse_conv_and_bn

            self.mlp[0] = fuse_conv_and_bn(self.mlp[0], self.mlp[1])
            self.mlp[1] = nn.Identity()

    def forward(self, x):
        x = self.adjust(x)
        shortcut = x
        x = self.mlp(self.pconv(x))
        if self.layer_scale is not None:
            x = x * self.layer_scale.view(1, -1, 1, 1)
        return shortcut + self.drop_path(x)


class C3k_Faster(C3k):
    """C3k with FasterNet blocks replacing the standard Bottlenecks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FasterNetBlock(c_) for _ in range(n)))


class C3k2_Faster(C2f):
    """C3k2 with FasterNet blocks; drop-in for DSC3k2 / C3k2 in YOLOv13 YAMLs.

    YAML args map exactly like C3k2 / DSC3k2: [c2, c3k, e].
    c3k=False -> plain FasterNetBlocks; c3k=True -> stacked C3k_Faster blocks.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_Faster(self.c, self.c, 2, shortcut, g) if c3k else FasterNetBlock(self.c)
            for _ in range(n)
        )


class TConv(nn.Module):
    """T-shaped replacement for a dense k x k stride-1 convolution (Sec. 3.3).

    PConv (spatial mixing on c1/n_div channels) followed by a dense PWConv
    (1x1 + BN + act) that mixes all channels and handles c1 -> c2. Their
    combined effective receptive field is T-shaped: dense over channels only
    at the center position. Stride-1 only: PConv's identity branch cannot
    downsample, so strided convs must stay regular (the paper likewise keeps
    plain convs for embedding/merging).
    """

    def __init__(self, c1, c2, k=PCONV_KERNEL, n_div=PCONV_N_DIV, act=True):
        super().__init__()
        self.pconv = PConv(c1, n_div, k)
        self.pw = Conv(c1, c2, 1, act=act)

    def forward(self, x):
        return self.pw(self.pconv(x))


class DetectFaster(Detect):
    """Detect head with the box branch (cv2) dense 3x3 convs replaced by TConv.

    Rationale (measured on yolov13n at 640): cv2's dense 3x3 stride-1 convs
    cost ~1.45 GFLOPs, the single largest remaining dense-conv group in the
    model. The classification branch (cv3) already uses DWConv pairs in the
    non-legacy layout and is left untouched, as are the final 1x1 output
    convs. Drop-in for Detect in any YAML: [..., 1, DetectFaster, [nc]].
    """

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        self.cv2 = nn.ModuleList(
            nn.Sequential(TConv(x, c2, 3), TConv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        if self.end2end:  # rebuild the one2one copy from the replaced branch
            self.one2one_cv2 = copy.deepcopy(self.cv2)
