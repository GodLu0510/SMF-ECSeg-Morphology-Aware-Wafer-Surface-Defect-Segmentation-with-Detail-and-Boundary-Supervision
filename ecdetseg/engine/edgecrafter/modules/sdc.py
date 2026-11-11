"""SMF-ECSeg paper module, extracted from the trained implementation.
See LICENSE.md and THIRD_PARTY_NOTICES.md.
"""
import torch
from torch import nn
import torch.nn.functional as F

from ..ecvit import ConvPyramidPatchEmbed, ViTAdapter


class WaferDetailAdapter(ViTAdapter):
    """Fuse a genuine stride-8 convolutional stem map into the semantic P3.

    Existing ViT parameter names are preserved. Extra modules do not advance the
    RNG used to initialize later baseline modules, enabling fair same-seed runs.
    """
    def __init__(self, embed_dim=192, detail_strength=0.1, **kwargs):
        super().__init__(embed_dim=embed_dim, **kwargs)
        if not isinstance(self.backbone.patch_embed, ConvPyramidPatchEmbed):
            raise ValueError('WaferDetailAdapter requires ConvPyramidPatchEmbed.')
        if len(self.projector) != 1 or self.projector[0].conv.out_channels != embed_dim:
            raise ValueError('WaferDetailAdapter requires the native equal-width pyramid.')
        with torch.random.fork_rng(devices=[]):
            self.detail_projection = nn.Sequential(
                nn.Conv2d(embed_dim // 2, embed_dim, 1, bias=False),
                nn.GroupNorm(8, embed_dim), nn.SiLU())
            self.detail_gate = nn.Conv2d(embed_dim * 2, embed_dim, 1)
            self.detail_scale = nn.Parameter(torch.full((1, embed_dim, 1, 1), float(detail_strength)))

    def forward(self, x):
        embedding, detail = self.backbone.patch_embed.forward_with_detail(x)
        layers = self.backbone.forward_from_embedding(embedding)
        semantic = torch.stack(layers).mean(0).transpose(1, 2).reshape(
            x.shape[0], -1, embedding.shape[-2], embedding.shape[-1]).contiguous()
        pyramid = [F.interpolate(semantic, size=(embedding.shape[-2] * 2 ** (1-i),
                    embedding.shape[-1] * 2 ** (1-i)), mode='bilinear', align_corners=False)
                   for i in range(2)]
        pyramid.append(F.interpolate(semantic, size=(embedding.shape[-2] // 2,
                       embedding.shape[-1] // 2), mode='bilinear', align_corners=False))
        pyramid[-1] = self.projector[-1](pyramid[-1])
        detail = self.detail_projection(detail)
        gate = torch.sigmoid(self.detail_gate(torch.cat((pyramid[0], detail), dim=1)))
        pyramid[0] = pyramid[0] + self.detail_scale * gate * detail
        return pyramid
