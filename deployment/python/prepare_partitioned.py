"""Prepare physically separate FP16 neural, FP32 geometry and cast ONNX engines."""
import argparse
import copy
import json
from pathlib import Path
from collections import Counter
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T
from ecseg_common import sha256, dump_json
DT={T.FLOAT:'float32',T.FLOAT16:'float16',T.INT64:'int32',T.INT32:'int32',T.BOOL:'bool'}


def make_parts(model,folder):
    info={v.name:v for v in [*model.graph.input,*model.graph.value_info,*model.graph.output]}
    inits={v.name:v for v in model.graph.initializer}
    final={v.name for v in model.graph.output}
    consumers={}
    for i,node in enumerate(model.graph.node):
        for name in node.input:consumers.setdefault(name,set()).add(i)
    def half(node):
        # Integral is a fixed distribution-to-distance projection, not a learned
        # neural layer. Its entire geometry path remains FP32.
        return node.op_type in ('MatMul','Gemm','Conv') and '/integral/' not in node.name
    available=set(inits)|{v.name for v in model.graph.input}
    pending=set(range(len(model.graph.node)));groups=[]
    color=False
    while pending:
        selected=[]
        for i,node in enumerate(model.graph.node):
            if i in pending and half(node)==color and all(not n or n in available for n in node.input):
                selected.append(i);pending.remove(i);available.update(node.output)
        if selected:groups.append((color,selected))
        else:
            # A topological node order lets one scan reach the fixed point for a
            # color; an empty pair of scans therefore means an invalid graph.
            if groups and groups[-1][0]!=color and not any(
                all(not n or n in available for n in model.graph.node[i].input) for i in pending):
                raise ValueError('No ready nodes in decoder DAG')
        color=not color
    buffers={};parts={};steps=[];resident={};cast_cache={}
    def spec(name,precision):
        v=info[name];tt=v.type.tensor_type;shape=[]
        for dim in tt.shape.dim:
            if not dim.HasField('dim_value'):raise ValueError('Non-static boundary: '+name)
            shape.append(dim.dim_value)
        dtype='float16' if precision and tt.elem_type==T.FLOAT else DT[tt.elem_type]
        return {'shape':shape,'dtype':dtype,'onnx_type':T.FLOAT16 if dtype=='float16' else tt.elem_type}
    def storage(name,dtype):
        key=name+'@@'+dtype
        if key not in buffers:
            value=spec(name,dtype=='float16')
            value.pop('onnx_type');value['bytes']=int(np.prod(value['shape']))*np.dtype(value['dtype']).itemsize
            buffers[key]=value
        return key
    for v in model.graph.input:
        key=storage(v.name,DT[v.type.tensor_type.elem_type]);resident[(v.name,DT[v.type.tensor_type.elem_type])]=key
    def need(name,precision):
        desired=spec(name,precision)['dtype'];key=(name,desired)
        if key in resident:return resident[key]
        sources=[(d,k) for (n,d),k in resident.items() if n==name]
        if len(sources)!=1 or sources[0][0] not in ('float16','float32') or desired not in ('float16','float32'):
            raise ValueError('Unsupported live value conversion: '+name)
        source_type,source_key=sources[0];outkey=storage(name,desired)
        count=int(np.prod(buffers[outkey]['shape']));signature=(count,source_type,desired)
        if signature not in cast_cache:
            part='cast_%03d'%len(cast_cache);cast_cache[signature]=part
            a=T.FLOAT16 if source_type=='float16' else T.FLOAT;b=T.FLOAT16 if desired=='float16' else T.FLOAT
            graph=H.make_graph([H.make_node('Cast',['input'],['output'],name=part,to=b)],part,
                [H.make_tensor_value_info('input',a,[count])],[H.make_tensor_value_info('output',b,[count])])
            m=H.make_model(graph,opset_imports=[H.make_opsetid('',13)],ir_version=7)
            onnx.checker.check_model(m);onnx.save(m,str(folder/(part+'.onnx')))
            parts[part]={'file':part+'.onnx','precision':'cast','source_dtype':source_type,'target_dtype':desired,
                'inputs':{'input':{'shape':[count],'dtype':source_type}},'outputs':{'output':{'shape':[count],'dtype':desired}},'nodes':1}
        steps.append({'part':cast_cache[signature],'inputs':{'input':source_key},'outputs':{'output':outkey}})
        resident[key]=outkey
        return outkey
    neural_nodes=[]
    for number,(is_half,indices) in enumerate(groups):
        selected=set(indices);nodes=[copy.deepcopy(model.graph.node[i]) for i in indices]
        defined={n for node in nodes for n in node.output}
        input_names=list(dict.fromkeys(n for node in nodes for n in node.input if n and n not in defined and n not in inits))
        output_names=list(dict.fromkeys(n for node in nodes for n in node.output if n in final or consumers.get(n,set())-selected))
        used_inits=list(dict.fromkeys(n for node in nodes for n in node.input if n in inits))
        weights=[]
        for name in used_inits:
            tensor=copy.deepcopy(inits[name])
            if is_half and tensor.data_type==T.FLOAT:
                array=N.to_array(tensor)
                with np.errstate(over='ignore'):rounded=array.astype(np.float16)
                if not np.isfinite(rounded).all():raise ValueError('Non-finite neural FP16 initializer: '+name)
                tensor=N.from_array(rounded,name)
            weights.append(tensor)
        def vi(name):
            s=spec(name,is_half)
            return H.make_tensor_value_info(name,s['onnx_type'],s['shape'])
        part=('neural16_' if is_half else 'control32_')+'%03d'%number
        graph=H.make_graph(nodes,part,[vi(n) for n in input_names],[vi(n) for n in output_names],weights)
        m=H.make_model(graph,opset_imports=[H.make_opsetid('',13)],ir_version=7)
        onnx.checker.check_model(m,full_check=True);onnx.save(m,str(folder/(part+'.onnx')))
        parts[part]={'file':part+'.onnx','precision':'fp16' if is_half else 'fp32','nodes':len(nodes),
            'inputs':{n:{k:v for k,v in spec(n,is_half).items() if k!='onnx_type'} for n in input_names},
            'outputs':{n:{k:v for k,v in spec(n,is_half).items() if k!='onnx_type'} for n in output_names}}
        ins={n:need(n,is_half) for n in input_names};outs={}
        for n in output_names:
            dtype=spec(n,is_half)['dtype'];key=storage(n,dtype);resident[(n,dtype)]=key;outs[n]=key
        steps.append({'part':part,'inputs':ins,'outputs':outs})
        if is_half:neural_nodes.extend({'name':n.name,'op':n.op_type} for n in nodes)
    outputs={v.name:need(v.name,False) for v in model.graph.output}
    for part in parts.values():part['sha256']=sha256(folder/part['file'])
    return {'parts':parts,'steps':steps,'buffers':buffers,'outputs':outputs,
        'inputs':{v.name:resident[(v.name,DT[v.type.tensor_type.elem_type])] for v in model.graph.input},
        'FP16_neural_nodes':neural_nodes,'physical_engine_counts':dict(Counter(p['precision'] for p in parts.values())),
        'policy':'Separate homogeneous FP16 neural engines, FP32 control/geometry engines, standalone type-conversion engines; all on GPU'}

def main():
    import onnxsim
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, help='TensorRT project containing model/decoder.onnx')
    args = parser.parse_args()
    source = Path(args.project).resolve() / 'model'
    folder = source / 'fp16_partition'
    folder.mkdir(parents=True, exist_ok=False)
    model, okay = onnxsim.simplify(onnx.load(str(source / 'decoder.onnx')), check_n=0)
    if not okay:
        raise RuntimeError('ONNX simplifier validation failed')
    plan = make_parts(model, folder)
    plan.update(source_decoder_sha256=sha256(source / 'decoder.onnx'),
                manifest_sha256=sha256(source / 'manifest.json'),
                schema='edgecrafter.tx2.physical_precision_partition.v1')
    dump_json(folder / 'partition.json', plan)
    print(json.dumps(plan['physical_engine_counts'], indent=2))
    print('Partitions prepared. TensorRT building and accuracy evaluation run on the target Jetson.')

if __name__ == '__main__':
    main()
