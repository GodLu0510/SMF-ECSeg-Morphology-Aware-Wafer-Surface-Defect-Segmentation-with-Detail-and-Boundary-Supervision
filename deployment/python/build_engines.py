#!/usr/bin/env python3
"""Build actual TensorRT 8.2 engines on TX2; no host-built engine is shipped."""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import platform
import sys
import time
import traceback

from ecseg_common import sha256,dump_json,load_manifest

def inspector(engine,trt):
    inspect=engine.create_engine_inspector()
    text=inspect.get_engine_information(trt.LayerInformationFormat.JSON)
    try:return json.loads(text)
    except ValueError:return {'raw':text}

def protect(layer,trt):
    """Keep normalization, softmax and bilinear-coordinate arithmetic in FP32."""
    name=layer.name
    floating=[i for i in range(layer.num_outputs) if layer.get_output(i) is not None and layer.get_output(i).dtype==trt.float32]
    if not floating:return False
    sensitive=('/TX2_FP32_NORM/' in name or '/TX2_FP32_SAMPLE/' in name
               or layer.type in (trt.LayerType.SOFTMAX,trt.LayerType.PLUGIN,trt.LayerType.PLUGIN_V2))
    arithmetic=(trt.LayerType.ELEMENTWISE,trt.LayerType.UNARY,trt.LayerType.REDUCE,
                trt.LayerType.SOFTMAX,trt.LayerType.PLUGIN,trt.LayerType.PLUGIN_V2)
    if sensitive and layer.type in arithmetic:
        layer.precision=trt.float32
        for i in floating:layer.set_output_type(i,trt.float32)
        return True
    return False

def decoder_precision_bridges(network,trt):
    """FP16 conv/FC/matmul with explicit casts to a FP32 decoder control path.

    Avoid implicit mixed-type Myelin residual fusions in TRT8.2. This is an
    explicitly disclosed mixed-precision policy, not an all-FP16 decoder.
    Integer indices, shape tensors and boolean masks retain their original types.
    """
    layers=[network.get_layer(i) for i in range(network.num_layers)]
    float_types=(trt.float32,trt.float16)
    learned_types=(trt.LayerType.CONVOLUTION,trt.LayerType.FULLY_CONNECTED,trt.LayerType.MATRIX_MULTIPLY)
    compute_types=learned_types+(trt.LayerType.ELEMENTWISE,trt.LayerType.UNARY,trt.LayerType.REDUCE,
        trt.LayerType.SOFTMAX,trt.LayerType.ACTIVATION,trt.LayerType.POOLING,
        trt.LayerType.PLUGIN,trt.LayerType.PLUGIN_V2)
    external_outputs={network.get_output(i).name for i in range(network.num_outputs)}
    contracts={network.get_input(i).name:trt.float32 for i in range(network.num_inputs)
               if network.get_input(i).dtype in float_types}
    policies=[];protected=[];half=[]
    for layer in layers:
        inputs=[layer.get_input(i) for i in range(layer.num_inputs)]
        outputs=[layer.get_output(i) for i in range(layer.num_outputs)]
        floating_outputs=[(i,t) for i,t in enumerate(outputs) if t is not None and t.dtype in float_types]
        floating_inputs=[(i,t) for i,t in enumerate(inputs) if t is not None and t.dtype in float_types
                         and not getattr(t,'is_shape_tensor',False)]
        if not floating_inputs and not floating_outputs:continue
        dtype=trt.float16 if layer.type in learned_types else trt.float32
        if dtype==trt.float16 and any(t.name in external_outputs for _,t in floating_outputs):
            raise ValueError('Unexpected direct FP16 learned-layer output binding: '+layer.name)
        policies.append((layer,dtype,floating_inputs,floating_outputs))
        for _,tensor in floating_outputs:contracts[tensor.name]=dtype
    for layer,dtype,inputs,outputs in policies:
        if outputs and layer.type in compute_types:layer.precision=dtype
        for i,tensor in outputs:layer.set_output_type(i,dtype)
        if dtype==trt.float16:half.append(layer.name)
        elif outputs:protected.append(layer.name)
    conversions=[];cache={}
    for layer,dtype,inputs,outputs in policies:
        for index,tensor in inputs:
            source_type=contracts.get(tensor.name,tensor.dtype)
            if source_type==dtype:continue
            key=(tensor.name,str(dtype))
            if key not in cache:
                cast=network.add_identity(tensor)
                if cast is None:raise RuntimeError('Failed to create explicit precision conversion')
                target='FP16' if dtype==trt.float16 else 'FP32'
                cast.name='/TX2_PRECISION_BRIDGE/%s_%04d'%(target,len(conversions))
                cast.set_output_type(0,dtype)
                output=cast.get_output(0)
                output.name=cast.name+'_output';output.dtype=dtype
                cache[key]=output
                conversions.append({'name':cast.name,'source_tensor':tensor.name,
                    'source_type':str(source_type),'target_type':str(dtype)})
            layer.set_input(index,cache[key])
    if not half:raise RuntimeError('Decoder FP16 policy did not select any convolution/matrix operations')
    return {'policy':'Explicit FP16 convolution/FC/matmul; FP32 remaining decoder path; no implicit mixed residual Add',
        'requested_FP16_compute_layers':half,'protected_FP32_layers':protected,'conversions':conversions,
        'actual_engine_precision':'See decoder_layers.json from actual target build; board AP pending'}


def build_one(root,precision,stage,args):
    import tensorrt as trt
    from tx2_runtime import Cuda
    manifest=load_manifest(root/'model/manifest.json')
    path=root/'model'/(stage+'.onnx')
    if sha256(path)!=manifest['onnx_files'][path.name]:raise ValueError('ONNX source changed')
    target=root/'engines'/precision
    target.mkdir(parents=True,exist_ok=True)
    engine_path=target/(stage+'.engine')
    record_path=target/(stage+'.json')
    device=Cuda().device_info()
    binding={'onnx_sha256':sha256(path),'manifest_sha256':sha256(root/'model/manifest.json'),
        'precision':precision,'tensorrt':trt.__version__,'GPU':device,'workspace_mib':args.workspace_mib,
        'policy':'FP32 only; FP16/INT8/TF32 disabled' if precision=='fp32' else
                 ('FP16 encoder with FP32 norm/softmax/bilinear coordinates' if stage=='encoder' else
                  'Explicit FP16 convolution/FC/matmul; FP32 remaining decoder path; conversions recorded; actual inspector saved'),
        'source_checkpoint_sha256':manifest['checkpoint_sha256'],
        'builder_source_sha256':sha256(__file__),'runtime_source_sha256':sha256(root/'tx2_runtime.py')}
    if engine_path.is_file() and record_path.is_file() and not args.rebuild:
        old=json.loads(record_path.read_text(encoding='utf-8'))
        if all(old.get(k)==v for k,v in binding.items()) and old.get('engine_sha256')==sha256(engine_path):
            print('Reusing verified engine:',engine_path,flush=True)
            return old
        raise ValueError('Existing engine has different bindings; choose --rebuild to replace it with a backup')
    logger=trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger,'')
    builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(path.read_bytes()):
        errors=[str(parser.get_error(i)) for i in range(parser.num_errors)]
        dump_json(target/(stage+'_parser_errors.json'),{'errors':errors,'binding':binding})
        raise RuntimeError('TensorRT ONNX parser failed:\n'+'\n'.join(errors))
    cfg=builder.create_builder_config()
    cfg.max_workspace_size=args.workspace_mib*1024*1024
    cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    for flag in ('FP16','INT8','TF32'):
        if hasattr(trt.BuilderFlag,flag):cfg.clear_flag(getattr(trt.BuilderFlag,flag))
    protected=[];precision_bridges=None
    if precision=='fp16':
        if not builder.platform_has_fast_fp16:raise RuntimeError('GPU does not report fast FP16 support')
        cfg.set_flag(trt.BuilderFlag.FP16)
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        if stage=='decoder':
            precision_bridges=decoder_precision_bridges(network,trt)
            protected=precision_bridges['protected_FP32_layers']
            print('Decoder precision bridges: %d; requested FP16 compute layers: %d'%(
                len(precision_bridges['conversions']),len(precision_bridges['requested_FP16_compute_layers'])),flush=True)
        else:
            for i in range(network.num_layers):
                layer=network.get_layer(i)
                if protect(layer,trt):protected.append(layer.name)
    inputs={network.get_input(i).name:list(network.get_input(i).shape) for i in range(network.num_inputs)}
    outputs={network.get_output(i).name:list(network.get_output(i).shape) for i in range(network.num_outputs)}
    expected_inputs={'images':[1,3]+manifest['input_size']} if stage=='encoder' else dict(zip(manifest['feature_names'],manifest['feature_shapes']))
    expected_outputs=dict(zip(manifest['feature_names'],manifest['feature_shapes'])) if stage=='encoder' else dict(zip(manifest['output_names'],manifest['output_shapes']))
    if inputs!=expected_inputs or outputs!=expected_outputs:
        raise ValueError('TensorRT graph I/O differs from fixed export contract: '+str((inputs,outputs)))
    for tensor in [network.get_input(i) for i in range(network.num_inputs)]+[network.get_output(i) for i in range(network.num_outputs)]:
        tensor.dtype=trt.float32
        tensor.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
    dump_json(target/(stage+'_build_plan.json'),dict(binding,inputs=inputs,outputs=outputs,
        protected_FP32_layers=protected,precision_bridges=precision_bridges))
    print('Building %s %s on actual GPU; workspace=%s MiB'%(precision,stage,args.workspace_mib),flush=True)
    begin=time.monotonic()
    serialized=builder.build_serialized_network(network,cfg)
    if serialized is None:raise RuntimeError('TensorRT build returned no serialized engine; inspect this build log')
    runtime=trt.Runtime(logger)
    engine=runtime.deserialize_cuda_engine(serialized)
    if engine is None:raise RuntimeError('Built engine cannot be deserialized')
    details=inspector(engine,trt)
    metadata=dict(binding,build_seconds=time.monotonic()-begin,protected_FP32_layers=protected,precision_bridges=precision_bridges,
        device_memory_bytes=engine.device_memory_size,bindings=[{'name':engine.get_binding_name(i),
            'input':engine.binding_is_input(i),'shape':list(engine.get_binding_shape(i)),
            'dtype':str(engine.get_binding_dtype(i))} for i in range(engine.num_bindings)])
    if any(r['dtype']!='DataType.FLOAT' for r in metadata['bindings']):raise ValueError('Expected FP32 external engine bindings')
    if engine_path.exists():
        backup=engine_path.with_name(stage+'.'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.backup.engine')
        engine_path.replace(backup)
    temporary=engine_path.with_suffix('.engine.tmp')
    temporary.write_bytes(bytes(serialized))
    temporary.replace(engine_path)
    metadata.update(engine_sha256=sha256(engine_path),engine_bytes=engine_path.stat().st_size,
        status='built_on_target; inference/AP verification pending')
    dump_json(target/(stage+'_layers.json'),details)
    dump_json(record_path,metadata)
    del engine,runtime,serialized,parser,network,cfg,builder
    gc.collect()
    return metadata

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--precision',choices=('fp32','fp16','both'),default='both')
    parser.add_argument('--stage',choices=('encoder','decoder','both'),default='both')
    parser.add_argument('--workspace-mib',type=int,default=512)
    parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args()
    if platform.system()!='Linux' or platform.machine().lower() not in ('aarch64','arm64'):
        parser.error('Build on Jetson Linux ARM64, not the Windows/Ubuntu x86 host')
    import tensorrt as trt
    if not trt.__version__.startswith('8.2.'):
        parser.error('This delivery targets the confirmed TensorRT 8.2 runtime')
    if args.workspace_mib<64:parser.error('Workspace must be at least 64 MiB')
    root=Path(args.project).resolve()
    try:
        for precision in (('fp32','fp16') if args.precision=='both' else (args.precision,)):
            for stage in (('encoder','decoder') if args.stage=='both' else (args.stage,)):
                build_one(root,precision,stage,args)
    except Exception:
        error={'status':'build_failed','traceback':traceback.format_exc()}
        dump_json(root/'build_error.json',error)
        raise
    print('Requested engines built. See docs/DEPLOYMENT.md for inference commands.', flush=True)

if __name__=='__main__':main()
