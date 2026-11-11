"""Dynamic/static FP32 ONNX export for JetPack 4.6 ONNX Runtime."""
import argparse
import copy
from pathlib import Path
import sys
from types import MethodType
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torch import nn
from ecseg_common import preprocess, sha256, dump_json, OUTPUT_NAMES, load_manifest
from ecseg_export import working_directory


class Encoder(nn.Module):
    def __init__(self,graph):super().__init__();self.graph=graph
    def forward(self,images):
        values=[]
        for index,(sources,path,stage) in enumerate(self.graph._nodes):
            get=lambda item:self.graph._input(item,index,values,images)
            inputs=[get(item) for item in sources] if isinstance(sources,list) else get(sources)
            if stage=='decoder':return tuple(inputs)
            values.append(self.graph.get_submodule(path)(inputs))


class Decoder(nn.Module):
    def __init__(self,head):super().__init__();self.head=head
    def forward(self,p3,p4,p5):
        result=self.head([p3,p4,p5],spatial_feat=p3)
        return tuple(result[n] for n in OUTPUT_NAMES)


def enable_dynamic_spatial(graph):
    """Preserve learned tensors; replace shape freezing/cached-640 computations."""
    from engine.edgecrafter.graph_model import TransformerEncoder2D
    from engine.edgecrafter.hybrid_encoder import HybridEncoder
    def positions(w,h,embed_dim=256,temperature=10000.):
        grid_w=torch.arange(w,dtype=torch.float32);grid_h=torch.arange(h,dtype=torch.float32)
        grid_w,grid_h=torch.meshgrid(grid_w,grid_h,indexing='ij')
        pos_dim=embed_dim//4;omega=1./(temperature**(torch.arange(pos_dim,dtype=torch.float32)/pos_dim))
        out_w=grid_w.flatten()[...,None]@omega[None];out_h=grid_h.flatten()[...,None]@omega[None]
        return torch.cat([out_w.sin(),out_w.cos(),out_h.sin(),out_h.cos()],dim=1)[None,:,:]
    HybridEncoder.build_2d_sincos_position_embedding=staticmethod(positions)
    for module in graph.modules():
        if isinstance(module,TransformerEncoder2D):module.pos_embed=None
    head=graph.decoder;head.eval_spatial_size=None
    def anchors(self,spatial_shapes=None,grid_size=.05,dtype=torch.float32,device='cpu'):
        if spatial_shapes is None:raise ValueError('Dynamic anchors require actual feature shapes')
        values=[]
        for level,(h,w) in enumerate(spatial_shapes):
            gy=(torch.arange(h,dtype=dtype,device=device)+.5)/h
            gx=(torch.arange(w,dtype=dtype,device=device)+.5)/w
            yy,xx=torch.meshgrid(gy,gx,indexing='ij');xy=torch.stack([xx,yy],dim=-1)
            wh=torch.ones_like(xy)*(grid_size*(2.**level))
            values.append(torch.cat([xy,wh],dim=-1).reshape(1,-1,4))
        a=torch.cat(values,dim=1);valid=((a>self.eps)*(a<1-self.eps)).all(-1,keepdim=True)
        return torch.where(valid,torch.log(a/(1-a)),torch.inf),valid
    head._generate_anchors=MethodType(anchors,head)
    seg=head.decoder.segmentation_head
    ratio=head.feat_strides[0]//seg.downsample_ratio
    assert ratio==2
    def masks(self,spatial_features,query_features,skip_blocks=False):
        target=(spatial_features.shape[-2]*ratio,spatial_features.shape[-1]*ratio)
        spatial_features=torch.nn.functional.interpolate(spatial_features,size=target,mode='bilinear',align_corners=False)
        result=[]
        if not skip_blocks:
            for block,qf in zip(self.blocks,query_features):
                spatial_features=block(spatial_features)
                projected=self.spatial_features_proj(spatial_features)
                q=self.query_features_proj(self.query_features_block(qf))
                result.append(torch.einsum('bchw,bnc->bnhw',projected,q)+self.bias)
        else:
            assert len(query_features)==1
            q=self.query_features_proj(self.query_features_block(query_features[0]))
            result.append(torch.einsum('bchw,bnc->bnhw',spatial_features,q)+self.bias)
        return result
    seg.forward=MethodType(masks,seg)


def sampler_domain(model):
    model=copy.deepcopy(model);count=0
    for node in model.graph.node:
        if node.op_type=='GridSample' and not node.domain:
            attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
            assert attrs.get('mode',b'bilinear')==b'bilinear' and attrs.get('padding_mode',b'zeros')==b'zeros'
            assert attrs.get('align_corners',0)==0
            node.domain='com.microsoft';count+=1
    if count and not any(o.domain=='com.microsoft' for o in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid('com.microsoft',1))
    model.ir_version=8
    assert not any(n.op_type=='LayerNormalization' for n in model.graph.node)
    onnx.checker.check_model(model)
    return model,count

def main():
    import onnxsim
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='Training ecdetseg directory')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True, help='Manifest from the original hybrid ONNX export')
    p.add_argument('--verify-image', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    source = Path(a.source).resolve(); config = Path(a.config).resolve()
    checkpoint = Path(a.checkpoint).resolve(); output = Path(a.output).resolve()
    base = load_manifest(a.manifest)
    if base['route'] != 'hybrid' or sha256(checkpoint) != base['checkpoint_sha256']:
        raise ValueError('Checkpoint and hybrid source manifest must match')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(a.threads); torch.backends.mha.set_fastpath_enabled(False)
    sys.path.insert(0, str(source))
    from engine.core import YAMLConfig
    with working_directory(source):
        cfg = YAMLConfig(str(config)); cfg.yaml_cfg.setdefault('ECGraph', {})['skip_load_backbone'] = True
        state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
        graph = cfg.model.cpu().eval(); graph.load_state_dict(state['ema']['module'], strict=True); graph.deploy()
        enable_dynamic_spatial(graph)
        encoder, decoder = Encoder(graph).eval(), Decoder(graph.decoder).eval()
        with Image.open(a.verify_image) as image:
            data = preprocess(image, base['input_size'])[1]
        sample = torch.from_numpy(data)
        with torch.no_grad():
            features = encoder(sample)
            expected = decoder(*features)
        dynamic = output / 'dynamic'; dynamic.mkdir()
        static = output / 'static'; static.mkdir()
        for name, module, inputs, names, outs in (
            ('encoder', encoder, (sample,), ['images'], base['feature_names']),
            ('decoder', decoder, features, base['feature_names'], OUTPUT_NAMES)):
            axes = {'images': {2: 'height', 3: 'width'}, 'p3': {2: 'p3_height', 3: 'p3_width'},
                    'p4': {2: 'p4_height', 3: 'p4_width'}, 'p5': {2: 'p5_height', 3: 'p5_width'},
                    'pred_masks': {2: 'mask_height', 3: 'mask_width'}}
            with torch.no_grad():
                torch.onnx.export(module, inputs, str(dynamic / (name + '.onnx')), opset_version=16,
                                  dynamo=False, do_constant_folding=True, input_names=names, output_names=outs,
                                  dynamic_axes={n: axes[n] for n in names + outs if n in axes})
            model, count = sampler_domain(onnx.load(str(dynamic / (name + '.onnx'))))
            onnx.save(model, str(dynamic / (name + '.onnx')))
            shapes = dict(zip(names, [list(x.shape) for x in inputs]))
            fixed, okay = onnxsim.simplify(model, overwrite_input_shapes=shapes, check_n=0)
            if not okay:
                raise RuntimeError('Static specialization failed: ' + name)
            onnx.save(fixed, str(static / (name + '.onnx')))
        options = ort.SessionOptions(); options.intra_op_num_threads = a.threads
        for mode, folder in [('dynamic', dynamic), ('static', static)]:
            enc = ort.InferenceSession(str(folder / 'encoder.onnx'), options, providers=['CPUExecutionProvider'])
            dec = ort.InferenceSession(str(folder / 'decoder.onnx'), options, providers=['CPUExecutionProvider'])
            values = dec.run(OUTPUT_NAMES, dict(zip(base['feature_names'], enc.run(None, {'images': data}))))
            errors = {}
            for name, value, target in zip(OUTPUT_NAMES, values, expected):
                reference = target.detach().numpy()
                if not np.isfinite(value).all():
                    raise ValueError('Non-finite ONNX output: ' + name)
                np.testing.assert_allclose(value, reference, rtol=6e-3, atol=6e-3)
                errors[name] = float(np.max(np.abs(value - reference)))
            manifest = copy.deepcopy(base)
            manifest.update(precision='FP32', shape_mode=mode, opset=16,
                            onnx_files={n: sha256(folder / n) for n in ('encoder.onnx', 'decoder.onnx')})
            dump_json(folder / 'manifest.json', manifest)
            dump_json(folder / 'verification.json', {'scope': 'CPU ONNX numerical check on the user-provided image',
                                                     'max_abs_errors': errors})
    print('Dynamic and static FP32 ONNX exports saved to:', output)

if __name__ == '__main__':
    main()
