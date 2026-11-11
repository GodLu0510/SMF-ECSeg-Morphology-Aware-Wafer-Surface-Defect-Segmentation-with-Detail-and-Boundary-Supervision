"""SMF-ECSeg paper module, extracted from the trained implementation.
See LICENSE.md and THIRD_PARTY_NOTICES.md.
"""
import torch
from torch import nn
import torch.nn.functional as F

from ..hybrid_encoder import RepNCSPELAN4


class WaferMorphologyBlock(RepNCSPELAN4):
    """Native P3 fusion plus content-adaptive local/strip/context branches.

    A semantic gate uses a local residual x-mean_neighborhood(x); no hard-coded
    edge threshold is used, since wafer circuit edges can also be strong.
    """
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act='silu', csp_type='csp2',
                 reduction=4, strip_kernel=7, context_kernel=5, residual_strength=0.1):
        super().__init__(c1, c2, c3, c4, n, bias, act, csp_type)
        if strip_kernel < 3 or strip_kernel % 2 != 1 or context_kernel % 2 != 1:
            raise ValueError('Strip/context kernels must be odd; strip kernel >= 3.')
        hidden = max(8, c2 // reduction)
        self.context_kernel = context_kernel
        with torch.random.fork_rng(devices=[]):
            self.morph_reduce = nn.Sequential(nn.Conv2d(c2, hidden, 1, bias=False),
                                             nn.GroupNorm(8, hidden), nn.SiLU())
            self.morph_branches = nn.ModuleList([
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                nn.Conv2d(hidden, hidden, (1, strip_kernel), padding=(0, strip_kernel//2), groups=hidden),
                nn.Conv2d(hidden, hidden, (strip_kernel, 1), padding=(strip_kernel//2, 0), groups=hidden),
                nn.Conv2d(hidden, hidden, 3, dilation=2, padding=2, groups=hidden)])
            self.morph_selector = nn.Conv2d(hidden * 2, 4, 1)
            self.morph_gate = nn.Conv2d(hidden * 2, hidden, 1)
            self.morph_expand = nn.Conv2d(hidden, c2, 1, bias=False)
            self.morph_scale = nn.Parameter(torch.full((1, c2, 1, 1), float(residual_strength)))

    def forward(self, x):
        original = super().forward(x)
        local = self.morph_reduce(original)
        residual = local - F.avg_pool2d(local, self.context_kernel, stride=1,
                                       padding=self.context_kernel//2, count_include_pad=False)
        context = torch.cat((local, residual), dim=1)
        weights = self.morph_selector(context).softmax(dim=1)
        mixed = sum(branch(local) * weights[:, i:i+1] for i, branch in enumerate(self.morph_branches))
        gated = torch.sigmoid(self.morph_gate(context)) * F.silu(mixed)
        return original + self.morph_scale * self.morph_expand(gated)
