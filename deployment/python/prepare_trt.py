"""Offline static opset-13 adaptation for the actual ECSeg TX2 TensorRT 8.2 graph.

Expand LayerNormalization and zero-padded bilinear GridSample, preserving the
original weights and RGB preprocessing. No TensorRT/board success is inferred.
"""
import copy
import json
from pathlib import Path
from collections import Counter
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T

class Expand:
    def __init__(self,graph,prefix):
        self.graph,self.prefix,self.nodes=graph,prefix,[]
        self.counter=0
    def const(self,name,value,dtype=np.float32):
        key=self.prefix+'/'+name
        self.graph.initializer.append(N.from_array(np.asarray(value,dtype=dtype),key))
        return key
    def op(self,kind,inputs,name=None,output=None,**attrs):
        self.counter+=1
        label=self.prefix+'/'+(name or (kind+'_'+str(self.counter)))
        output=output or label+'_output'
        self.nodes.append(H.make_node(kind,inputs,[output],name=label,**attrs))
        return output

def layer_norm(node,graph,shapes):
    attrs={a.name:H.get_attribute_value(a) for a in node.attribute}
    rank=len(shapes[node.input[0]])
    axis=attrs.get('axis',-1)%rank
    if len(node.output)!=1 or attrs.get('stash_type',1)!=1:
        raise ValueError('Only FP32-statistics, single-output LayerNormalization is supported')
    e=Expand(graph,node.name+'/TX2_FP32_NORM')
    x,scale,bias=node.input
    mean=e.op('ReduceMean',[x],axes=list(range(axis,rank)),keepdims=1)
    centered=e.op('Sub',[x,mean])
    square=e.op('Mul',[centered,centered])
    var=e.op('ReduceMean',[square],axes=list(range(axis,rank)),keepdims=1)
    eps=e.const('epsilon',attrs.get('epsilon',1e-5))
    denom=e.op('Sqrt',[e.op('Add',[var,eps])])
    normal=e.op('Div',[centered,denom])
    e.op('Add',[e.op('Mul',[normal,scale]),bias],output=node.output[0])
    return e.nodes

def grid_sample(node,graph,shapes):
    attrs={a.name:H.get_attribute_value(a) for a in node.attribute}
    if attrs.get('align_corners',0)!=0 or attrs.get('mode',b'bilinear')!=b'bilinear' or attrs.get('padding_mode',b'zeros')!=b'zeros':
        raise ValueError('Only align_corners=False, zero-padded bilinear GridSample is supported')
    n,c,h,w=shapes[node.input[0]]
    if any(not isinstance(v,int) or v<=0 for v in (n,c,h,w)):
        raise ValueError('GridSample source must have fixed positive NCHW dimensions')
    e=Expand(graph,node.name+'/TX2_FP32_SAMPLE')
    x,grid=node.input
    zero=e.const('zero',0);one=e.const('one',1);half=e.const('half',.5)
    width=e.const('width',w);height=e.const('height',h)
    maxx=e.const('max_x',w-1);maxy=e.const('max_y',h-1)
    gx=e.op('Gather',[grid,e.const('grid_x',0,np.int64)],axis=3)
    gy=e.op('Gather',[grid,e.const('grid_y',1,np.int64)],axis=3)
    # ((g+1)*dimension-1)/2, identical to ONNX align_corners=False.
    px=e.op('Mul',[e.op('Sub',[e.op('Mul',[e.op('Add',[gx,one]),width]),one]),half])
    py=e.op('Mul',[e.op('Sub',[e.op('Mul',[e.op('Add',[gy,one]),height]),one]),half])
    x0=e.op('Floor',[px]);y0=e.op('Floor',[py])
    x1=e.op('Add',[x0,one]);y1=e.op('Add',[y0,one])
    wx0=e.op('Sub',[x1,px]);wx1=e.op('Sub',[px,x0])
    wy0=e.op('Sub',[y1,py]);wy1=e.op('Sub',[py,y0])
    data=e.op('Transpose',[x],perm=[0,2,3,1])
    data=e.op('Reshape',[data,e.const('source_flat_shape',[-1,c],np.int64)])
    batch=e.const('batch_offsets',np.arange(n,dtype=np.int64).reshape(n,1,1)*(h*w),np.int64)
    iw=e.const('index_width',w,np.int64)
    axes=e.const('unsqueeze_axis',[3],np.int64)
    corners=[]
    for xx,yy,wx,wy,label in ((x0,y0,wx0,wy0,'00'),(x1,y0,wx1,wy0,'10'),(x0,y1,wx0,wy1,'01'),(x1,y1,wx1,wy1,'11')):
        validx=e.op('And',[e.op('GreaterOrEqual',[xx,zero]),e.op('Less',[xx,width])])
        validy=e.op('And',[e.op('GreaterOrEqual',[yy,zero]),e.op('Less',[yy,height])])
        valid=e.op('Cast',[e.op('And',[validx,validy])],to=T.FLOAT)
        ix=e.op('Cast',[e.op('Clip',[xx,zero,maxx])],to=T.INT64)
        iy=e.op('Cast',[e.op('Clip',[yy,zero,maxy])],to=T.INT64)
        index=e.op('Add',[e.op('Add',[e.op('Mul',[iy,iw]),ix]),batch])
        gathered=e.op('Gather',[data,index],axis=0)
        weight=e.op('Unsqueeze',[e.op('Mul',[e.op('Mul',[wx,wy]),valid]),axes])
        corners.append(e.op('Mul',[gathered,weight],name='weighted_'+label))
    added=e.op('Add',[e.op('Add',[corners[0],corners[1]]),e.op('Add',[corners[2],corners[3]])])
    e.op('Transpose',[added],perm=[0,3,1,2],output=node.output[0])
    return e.nodes

def adapt(source):
    weights={t.name:t.SerializeToString() for t in source.graph.initializer}
    model=onnx.shape_inference.infer_shapes(copy.deepcopy(source),data_prop=True)
    shapes={v.name:[int(d.dim_value) if d.HasField('dim_value') else d.dim_param for d in v.type.tensor_type.shape.dim]
            for v in list(model.graph.input)+list(model.graph.value_info)+list(model.graph.output)}
    graph=model.graph
    nodes=[]
    replaced=Counter()
    for node in list(graph.node):
        if node.op_type=='LayerNormalization':
            nodes.extend(layer_norm(node,graph,shapes));replaced[node.op_type]+=1
        elif node.op_type=='GridSample':
            nodes.extend(grid_sample(node,graph,shapes));replaced[node.op_type]+=1
        elif node.op_type=='Einsum':
            equation=next(a.s for a in node.attribute if a.name=='equation')
            if equation!=b'bchw,bnc->bnhw':raise ValueError('Unexpected Einsum')
            e=Expand(graph,node.name+'/TX2_mask_projection')
            b,c,h,w=shapes[node.input[0]]
            flat=e.op('Reshape',[node.input[0],e.const('spatial_shape',[b,c,h*w],np.int64)])
            product=e.op('MatMul',[node.input[1],flat])
            e.op('Reshape',[product,e.const('mask_shape',[b,300,h,w],np.int64)],output=node.output[0])
            nodes.extend(e.nodes);replaced[node.op_type]+=1
        else:
            node=copy.deepcopy(node)
            if node.op_type=='Reshape':
                for a in list(node.attribute):
                    if a.name=='allowzero':
                        if a.i:raise ValueError('Unsupported allowzero=1')
                        node.attribute.remove(a)
            nodes.append(node)
    del graph.node[:];graph.node.extend(nodes)
    # The generic converter has no 14->13 adapter for Relu. Audit actual schemas
    # instead: after expanding the new operators and removing allowzero=0,
    # remaining FP32 operators retain their semantics. Reject attribute/default
    # changes rather than silently changing their interpretation.
    for node in graph.node:
        old=onnx.defs.get_schema(node.op_type,13,node.domain)
        current=onnx.defs.get_schema(node.op_type,17,node.domain)
        present={a.name for a in node.attribute}
        if present-set(old.attributes):raise ValueError('Newer attributes on '+node.name)
        for key in set(old.attributes)&set(current.attributes)-present:
            aa,bb=old.attributes[key].default_value,current.attributes[key].default_value
            if aa.SerializeToString()!=bb.SerializeToString():
                raise ValueError('Changed attribute default on '+node.name+':'+key)
    for item in model.opset_import:
        if item.domain=='':item.version=13
    model.ir_version=7
    model=onnx.shape_inference.infer_shapes(model,data_prop=True)
    onnx.checker.check_model(model)
    after={t.name:t.SerializeToString() for t in model.graph.initializer}
    if any(after.get(name)!=value for name,value in weights.items()):
        raise ValueError('An original initializer was changed')
    if any(n.op_type in ('LayerNormalization','GridSample','Einsum') for n in model.graph.node):
        raise ValueError('Unconverted operator')
    return model,dict(replaced)


def main():
    import argparse
    import shutil
    from ecseg_common import load_manifest, sha256, dump_json
    from lqe_rewrite import rewrite_lqe
    parser = argparse.ArgumentParser(description='Adapt exported encoder/decoder ONNX to TensorRT 8.2.')
    parser.add_argument('--source', required=True, help='Directory containing original manifest.json and ONNX files')
    parser.add_argument('--project', required=True, help='New output project; models go under model/')
    args = parser.parse_args()
    source = Path(args.source).resolve()
    manifest = load_manifest(source / 'manifest.json')
    if manifest['route'] != 'hybrid':
        parser.error('TensorRT adaptation requires the hybrid encoder/decoder export')
    target = Path(args.project).resolve() / 'model'
    target.mkdir(parents=True, exist_ok=False)
    audit = {}
    for name in ('encoder', 'decoder'):
        original = source / (name + '.onnx')
        if sha256(original) != manifest['onnx_files'][original.name]:
            raise ValueError('Source ONNX checksum mismatch: ' + name)
        model, counts = adapt(onnx.load(str(original)))
        if name == 'decoder':
            model, changes = rewrite_lqe(model)
        else:
            changes = []
        onnx.checker.check_model(model)
        onnx.save(model, str(target / original.name))
        audit[name] = {'expanded': counts, 'lqe_rewrite': changes}
    manifest['onnx_files'] = {name + '.onnx': sha256(target / (name + '.onnx')) for name in ('encoder', 'decoder')}
    manifest.update(opset=13, precision='FP32', tx2_adaptation=audit)
    dump_json(target / 'manifest.json', manifest)
    dump_json(target / 'adaptation.json', audit)
    (target / 'classes.txt').write_text('\n'.join(manifest['classes']) + '\n', encoding='utf-8')
    print('Adapted ONNX saved. Build and evaluate TensorRT engines on the target Jetson.')

if __name__ == '__main__':
    main()
