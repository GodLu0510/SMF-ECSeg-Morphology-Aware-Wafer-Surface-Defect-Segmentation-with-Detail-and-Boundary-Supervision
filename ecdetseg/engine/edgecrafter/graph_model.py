"""Build EdgeCrafter from executable [from, module, args] YAML graphs."""
import copy
import inspect
from pathlib import Path
import re

import torch
from torch import nn
import yaml

from ..core import register
from ..core.workspace import GLOBAL_CONFIG
from .decoder import ECTransformer
from .ecvit import ViTAdapter
from .wafer_modules import WaferDetailAdapter, WaferMorphologyBlock
from .hybrid_encoder import (
    ConvNormLayer_fuse, HybridEncoder, RepNCSPELAN4, SCDown,
    TransformerEncoder, TransformerEncoderLayer,
)


class SelectFeature(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, features):
        return features[self.index]


class Add(nn.Module):
    def forward(self, features):
        result = features[0]
        for feature in features[1:]:
            result = result + feature
        return result


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, features):
        return torch.cat(features, dim=self.dim)


class TransformerEncoder2D(TransformerEncoder):
    """The native HybridEncoder attention operation, including its 2D positions.

    Inherits the native parameter layout (layers.*) for checkpoint compatibility.
    """
    __share__ = ['eval_spatial_size']

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.,
                 activation='silu', num_layers=1, pe_temperature=10000,
                 feat_stride=32, eval_spatial_size=None):
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        super().__init__(layer, num_layers)
        self.hidden_dim = d_model
        self.pe_temperature = pe_temperature
        self.pos_embed = None
        if eval_spatial_size is not None:
            self.pos_embed = HybridEncoder.build_2d_sincos_position_embedding(
                eval_spatial_size[1] // feat_stride, eval_spatial_size[0] // feat_stride,
                d_model, pe_temperature)

    def forward(self, feature):
        batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).permute(0, 2, 1)
        if self.training or self.pos_embed is None:
            positions = HybridEncoder.build_2d_sincos_position_embedding(
                width, height, self.hidden_dim, self.pe_temperature)
        else:
            positions = self.pos_embed
        memory = super().forward(tokens, pos_embed=positions.to(tokens.device))
        return memory.permute(0, 2, 1).reshape(batch, channels, height, width).contiguous()


_LOCAL_MODULES = {
    cls.__name__: cls for cls in (
        ViTAdapter, ECTransformer, HybridEncoder, ConvNormLayer_fuse,
        SCDown, RepNCSPELAN4, TransformerEncoder2D, SelectFeature, Add, Concat,
        WaferDetailAdapter, WaferMorphologyBlock,
    )
}


def resolve_graph_module(name):
    """Resolve existing classes without eval or executing YAML expressions."""
    if name in _LOCAL_MODULES:
        return _LOCAL_MODULES[name]
    if name.startswith('nn.'):
        candidate = getattr(nn, name[3:], None)
    else:
        schema = GLOBAL_CONFIG.get(name)
        candidate = getattr(schema['_pymodule'], schema['_name'], None) if isinstance(schema, dict) and '_pymodule' in schema else None
    if not inspect.isclass(candidate) or not issubclass(candidate, nn.Module):
        raise ValueError(f'Unknown graph module: {name}. Implement and register the module first.')
    if getattr(candidate, '__inject__', []):
        raise ValueError(f'{name} requires dependency injection; use its component modules in the graph.')
    return candidate


@register()
class ECGraph(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self, yaml_path, num_classes=80, eval_spatial_size=(640, 640),
                 skip_load_backbone=False):
        super().__init__()
        self.yaml_path = str(Path(yaml_path).resolve())
        with open(self.yaml_path, encoding='utf-8-sig') as stream:
            graph = yaml.safe_load(stream)
        if not isinstance(graph, dict) or graph.get('task') not in ('detection', 'segmentation'):
            raise ValueError('Graph task must be detection or segmentation.')
        self.task = graph['task']
        self.graph = copy.deepcopy(graph)
        self._nodes = []
        self._leaf_paths = set()
        shared = {'num_classes': num_classes, 'eval_spatial_size': eval_spatial_size}
        rows = []
        for stage in ('backbone', 'encoder', 'decoder'):
            definitions = graph.get(stage)
            if not isinstance(definitions, list) or not definitions:
                raise ValueError(f'Graph section {stage} must be a non-empty layer list.')
            if stage == 'decoder' and len(definitions) != 1:
                raise ValueError('Use one final ECTransformer decoder, including its integrated segmentation head.')
            rows.extend((stage, definition) for definition in definitions)
        paths = graph.get('parameter_paths', {})
        if not isinstance(paths, dict) or any(type(key) is not int or not 0 <= key < len(rows) for key in paths):
            raise ValueError('parameter_paths must map valid integer layer indices to parameter paths.')
        # Validate the complete topology before constructing modules or downloading weights.
        for index, (stage, row) in enumerate(rows):
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f'Layer {index}: expected [from, module, args].')
            source, name, args = row
            if not isinstance(name, str) or not isinstance(args, (dict, list)):
                raise ValueError(f'Layer {index}: module must be a name and args a dictionary or list.')
            sources = source if isinstance(source, list) else [source]
            if not sources or any(type(item) is not int for item in sources):
                raise ValueError(f'Layer {index}: from must be an integer or non-empty integer list.')
            for item in sources:
                previous = index + item if item < 0 else item
                if not (index == 0 and item == -1) and not 0 <= previous < index:
                    raise ValueError(f'Layer {index}: invalid or forward reference {item}.')
            module_type = resolve_graph_module(name)
            if stage == 'decoder' and not issubclass(module_type, ECTransformer):
                raise ValueError('The final decoder must be ECTransformer or its subclass.')
        for index, (stage, row) in enumerate(rows):
            source, name, args = row
            module_type = resolve_graph_module(name)
            try:
                arguments = copy.deepcopy(args) if isinstance(args, dict) else dict(inspect.signature(module_type).bind(*args).arguments)
                for key in getattr(module_type, '__share__', []):
                    if key in shared:
                        arguments[key] = shared[key]
                if issubclass(module_type, ViTAdapter) and skip_load_backbone:
                    arguments['skip_load_backbone'] = True
                if stage == 'decoder':
                    has_masks = bool(arguments.get('mask_downsample_ratio'))
                    if has_masks != (self.task == 'segmentation'):
                        raise ValueError('Segmentation requires mask_downsample_ratio; detection must omit it.')
                module = module_type(**arguments)
            except (TypeError, ValueError) as error:
                raise ValueError(f'Layer {index} ({name}): {error}') from error
            default_path = 'decoder' if stage == 'decoder' else f'graph_layers.{index}'
            module_path = paths.get(index, default_path)
            self._install_module(module_path, module)
            self._nodes.append((source, module_path, stage))
        final_sources = self._nodes[-1][0]
        if not isinstance(final_sources, list) or len(final_sources) != self.decoder.num_levels:
            raise ValueError('Decoder from must list one feature layer for each num_levels entry.')

    def _install_module(self, path, module):
        if not isinstance(path, str) or not re.fullmatch(r'[A-Za-z]\w*(?:\.(?:[A-Za-z]\w*|\d+))*', path):
            raise ValueError(f'Invalid parameter path: {path!r}')
        if any(path == leaf or path.startswith(leaf + '.') or leaf.startswith(path + '.') for leaf in self._leaf_paths):
            raise ValueError(f'Overlapping parameter path: {path}')
        parent = self
        parts = path.split('.')
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
            if not isinstance(parent, nn.Module):
                raise ValueError(f'Parameter path conflicts with a non-module attribute: {path}')
        if hasattr(parent, parts[-1]):
            raise ValueError(f'Parameter path already exists: {path}')
        parent.add_module(parts[-1], module)
        self._leaf_paths.add(path)

    @staticmethod
    def _input(source, index, values, image):
        if index == 0 and source == -1:
            return image
        return values[index + source if source < 0 else source]

    def forward(self, x, targets=None):
        values = []
        for index, (source, path, stage) in enumerate(self._nodes):
            inputs = [self._input(item, index, values, x) for item in source] if isinstance(source, list) else self._input(source, index, values, x)
            module = self.get_submodule(path)
            if stage == 'decoder':
                return module(inputs, targets, inputs[0] if self.task == 'segmentation' else None)
            values.append(module(inputs))

    def deploy(self):
        self.eval()
        for module in self.modules():
            if hasattr(module, 'convert_to_deploy'):
                module.convert_to_deploy()
        return self
