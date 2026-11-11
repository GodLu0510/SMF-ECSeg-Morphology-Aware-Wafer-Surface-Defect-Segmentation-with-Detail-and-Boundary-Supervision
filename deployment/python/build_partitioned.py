#!/usr/bin/env python3
"""Build/cache isolated FP16, FP32 and Cast engines on the actual TX2 GPU."""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import platform
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True);parser.add_argument('--rebuild',action='store_true')
    parser.add_argument('--workspace-mib',type=int,default=256)
    args=parser.parse_args();project=Path(args.project).resolve()
    if platform.machine().lower() not in ('aarch64','arm64') or platform.system()!='Linux':
        parser.error('Build on the actual Jetson Linux ARM64 device')
    if args.workspace_mib<32:parser.error('Workspace must be at least 32 MiB')
    sys.path.insert(0,str(project))
    import tensorrt as trt
    from tx2_runtime import Cuda
    from ecseg_common import sha256,dump_json
    if not trt.__version__.startswith('8.2.'):parser.error('Confirmed TensorRT8.2 required')
    model=project/'model/fp16_partition';target=project/'engines/fp16_partition';target.mkdir(parents=True,exist_ok=True)
    plan=json.loads((model/'partition.json').read_text())
    if sha256(project/'model/manifest.json')!=plan['manifest_sha256'] or sha256(project/'model/decoder.onnx')!=plan['source_decoder_sha256']:
        raise ValueError('Source model/manifest changed')
    encoder_path=project/'engines/fp16/encoder.engine'
    encoder=json.loads(encoder_path.with_suffix('.json').read_text())
    if (sha256(encoder_path)!=encoder['engine_sha256'] or encoder['manifest_sha256']!=plan['manifest_sha256']
        or encoder['precision']!='fp16' or encoder['onnx_sha256']!=sha256(project/'model/encoder.onnx')):
        raise ValueError('Existing FP16 encoder does not match this model')
    device=Cuda().device_info()
    if encoder['GPU']!=device or encoder['tensorrt']!=trt.__version__:raise ValueError('Encoder GPU/runtime binding differs')
    data_types={'float32':trt.float32,'float16':trt.float16,'int32':trt.int32,'bool':trt.bool}
    logger=trt.Logger(trt.Logger.INFO);trt.init_libnvinfer_plugins(logger,'')
    records={};active=None
    try:
        # Compile the formerly failing geometry and representative half/cast
        # islands early; a compiler incompatibility should not waste the entire
        # remaining build before it is reported. Execution order stays unchanged.
        order=list(plan['parts'])
        early=[n for n in order if plan['parts'][n]['precision']=='fp32'
               and b'/integral/' in (model/plan['parts'][n]['file']).read_bytes()]
        early+=next(([n] for n in order if plan['parts'][n]['precision']=='fp16'),[])
        early+=next(([n] for n in order if plan['parts'][n]['precision']=='cast'),[])
        order=list(dict.fromkeys(early+order))
        for number,part in enumerate(order):
            item=plan['parts'][part]
            active=part;file=model/item['file']
            if sha256(file)!=item['sha256']:raise ValueError('Partition ONNX changed: '+part)
            engine_path=target/(part+'.engine');record_path=target/(part+'.json')
            binding={'part':part,'onnx_sha256':item['sha256'],'plan_sha256':sha256(model/'partition.json'),
                'GPU':device,'tensorrt':trt.__version__,'precision':item['precision'],
                'builder_sha256':sha256(__file__),'workspace_mib':args.workspace_mib}
            if engine_path.exists() and record_path.exists() and not args.rebuild:
                old=json.loads(record_path.read_text())
                if all(old.get(k)==v for k,v in binding.items()) and old.get('engine_sha256')==sha256(engine_path):
                    records[part]=old;print('[%d/%d] Cached %s'%(number+1,len(plan['parts']),part),flush=True);continue
                raise ValueError('Cached part differs: '+part+'; use --rebuild to replace it with a backup')
            print('[%d/%d] Building %s (%s)'%(number+1,len(plan['parts']),part,item['precision']),flush=True)
            builder=trt.Builder(logger);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            importer=trt.OnnxParser(network,logger)
            if not importer.parse(file.read_bytes()):
                raise RuntimeError('\n'.join(str(importer.get_error(i)) for i in range(importer.num_errors)))
            config=builder.create_builder_config();config.max_workspace_size=args.workspace_mib*1024*1024
            config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
            for name in ('FP16','INT8','TF32'):
                if hasattr(trt.BuilderFlag,name):config.clear_flag(getattr(trt.BuilderFlag,name))
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            if item['precision'] in ('fp16','cast'):config.set_flag(trt.BuilderFlag.FP16)
            # Only network INPUT/OUTPUT types are set. No internal ITensor.dtype
            # setters, no in-engine mixed neural/normalization precision islands.
            for kind,count,getter in (('inputs',network.num_inputs,network.get_input),('outputs',network.num_outputs,network.get_output)):
                actual={getter(i).name:list(getter(i).shape) for i in range(count)}
                if actual!={n:v['shape'] for n,v in item[kind].items()}:raise ValueError('Static I/O mismatch: '+part)
                for i in range(count):
                    t=getter(i);t.dtype=data_types[item[kind][t.name]['dtype']]
                    t.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
            requested=[]
            if item['precision']=='fp16':
                compute=(trt.LayerType.MATRIX_MULTIPLY,trt.LayerType.FULLY_CONNECTED,trt.LayerType.CONVOLUTION,
                    trt.LayerType.ELEMENTWISE,trt.LayerType.SCALE)
                for i in range(network.num_layers):
                    layer=network.get_layer(i)
                    if layer.type in compute:
                        layer.precision=trt.float16
                        for j in range(layer.num_outputs):
                            t=layer.get_output(j)
                            if t.dtype in (trt.float32,trt.float16):layer.set_output_type(j,trt.float16)
                        requested.append(layer.name)
            start=time.monotonic();serialized=builder.build_serialized_network(network,config)
            if serialized is None:raise RuntimeError('No engine returned for '+part)
            runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(serialized)
            if engine is None:raise RuntimeError('Cannot deserialize '+part)
            rows=[]
            for i in range(engine.num_bindings):
                name=engine.get_binding_name(i);kind='inputs' if engine.binding_is_input(i) else 'outputs';expected=item[kind][name]
                if (list(engine.get_binding_shape(i))!=expected['shape'] or engine.get_binding_dtype(i)!=data_types[expected['dtype']]
                    or engine.get_binding_format(i)!=trt.TensorFormat.LINEAR or engine.is_shape_binding(i)):
                    raise ValueError('Unexpected actual binding/shape tensor in '+part+': '+name)
                rows.append({'name':name,'input':engine.binding_is_input(i),'shape':expected['shape'],'dtype':expected['dtype']})
            inspect=engine.create_engine_inspector();text=inspect.get_engine_information(trt.LayerInformationFormat.JSON)
            (target/(part+'_layers.json')).write_text(text,encoding='utf-8')
            if engine_path.exists():engine_path.replace(target/(part+'.'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.backup.engine'))
            temp=engine_path.with_suffix('.engine.tmp');temp.write_bytes(bytes(serialized));temp.replace(engine_path)
            record=dict(binding,engine_sha256=sha256(engine_path),engine_bytes=engine_path.stat().st_size,
                build_seconds=time.monotonic()-start,bindings=rows,requested_FP16_layers=requested,
                device_memory_bytes=engine.device_memory_size,status='built; board inference/AP pending')
            dump_json(record_path,record);records[part]=record
            del inspect,engine,runtime,serialized,importer,network,config,builder;gc.collect()
        dump_json(target/'index.json',{'status':'all_parts_built','parts':records,'encoder':encoder,
            'plan_sha256':sha256(model/'partition.json'),'GPU':device,'tensorrt':trt.__version__})
        print('All physical partitions built. Next: sudo bash run_fp16_partition.sh smoke',flush=True)
    except Exception:
        dump_json(target/'build_error.json',{'part':active,'completed_parts':list(records),'error':traceback.format_exc()})
        raise


if __name__=='__main__':main()
