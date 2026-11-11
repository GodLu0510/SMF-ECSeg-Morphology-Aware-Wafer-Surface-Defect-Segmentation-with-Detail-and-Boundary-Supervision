"""GPU-only physical precision partitions, using shared stream and scratch space."""
import ctypes as C
import json
from pathlib import Path
import time
import numpy as np
from ecseg_common import sha256
from tx2_runtime import Cuda


def memory_capsule(pointer):
    """TRT8.2's void* device_memory setter accepts a Python capsule."""
    create=C.pythonapi.PyCapsule_New
    create.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];create.restype=C.py_object
    return create(C.c_void_p(pointer),None,None)


def buffer_slots(plan):
    """Reuse device buffers only after the previous value's last queued read."""
    births={k:-1 for k in plan['inputs'].values()};ends={k:-1 for k in births}
    for i,step in enumerate(plan['steps']):
        for key in step['outputs'].values():births.setdefault(key,i);ends.setdefault(key,i)
        for key in step['inputs'].values():ends[key]=i
    for key in list(plan['inputs'].values())+list(plan['outputs'].values()):ends[key]=len(plan['steps'])
    assignments={};slots=[];live_ends=[]
    for key in sorted(births,key=lambda k:births[k]):
        size=max(256,int(plan['buffers'][key]['bytes']))
        candidates=[i for i,v in enumerate(slots) if live_ends[i]<births[key] and v>=size]
        slot=min(candidates,key=lambda i:slots[i]) if candidates else len(slots)
        if slot==len(slots):slots.append(size);live_ends.append(ends[key])
        else:live_ends[slot]=ends[key]
        assignments[key]=slot
    # A stage must never read an overwritten input buffer. Shared read-only
    # aliases are allowed only when they are literally the same logical value.
    for step in plan['steps']:
        keys=set(step['inputs'].values())|set(step['outputs'].values())
        if len({assignments[k] for k in keys})!=len(keys):raise ValueError('Overlapping live decoder buffers')
    return assignments,slots


class Engine:
    def __init__(self,args,manifest,out):
        import tensorrt as trt
        self.runtime=None;self.trt=trt;self.args=args;self.manifest=manifest
        self.cuda=Cuda();self.stream=C.c_void_p();self.events=[];self.devices=[];self.hosts=[]
        self.engines=[];self.contexts=[];self.runners=[];self.shared={};self.host_arrays={}
        root=Path(args.manifest).resolve().parent.parent
        self.model=root/'model/fp16_partition';self.directory=Path(args.engines).resolve()/'fp16_partition'
        self.plan=json.loads((self.model/'partition.json').read_text());self.index=json.loads((self.directory/'index.json').read_text())
        if args.precision!='fp16':raise ValueError('Partition runtime is for the explicitly named FP16 route')
        if (self.index['status']!='all_parts_built' or self.index['plan_sha256']!=sha256(self.model/'partition.json')
            or self.plan['manifest_sha256']!=sha256(args.manifest) or self.plan['source_decoder_sha256']!=sha256(root/'model/decoder.onnx')):
            raise ValueError('Physical precision plan/model/index binding mismatch')
        self.logger=trt.Logger(trt.Logger.WARNING);trt.init_libnvinfer_plugins(self.logger,'')
        self.trt_runtime=trt.Runtime(self.logger);self.cuda_info=self.cuda.device_info()
        try:
            if self.index['GPU']!=self.cuda_info or self.index['tensorrt']!=trt.__version__:raise ValueError('GPU/runtime mismatch')
            self.cuda.check(self.cuda.lib.cudaStreamCreate(C.byref(self.stream)),'partition stream')
            encoder_record=json.loads((Path(args.engines)/'fp16/encoder.json').read_text())
            if (encoder_record!=self.index['encoder'] or encoder_record['onnx_sha256']!=sha256(root/'model/encoder.onnx')
                or encoder_record['manifest_sha256']!=sha256(args.manifest)):
                raise ValueError('Existing encoder binding changed')
            encoder_path=Path(args.engines)/'fp16/encoder.engine'
            if sha256(encoder_path)!=encoder_record['engine_sha256']:raise ValueError('Encoder engine changed')
            encoder=self.trt_runtime.deserialize_cuda_engine(encoder_path.read_bytes())
            if encoder is None:raise RuntimeError('Cannot load encoder')
            self.engines.append(encoder)
            for part,p in self.plan['parts'].items():
                record=self.index['parts'][part];engine_path=self.directory/(part+'.engine')
                if (sha256(engine_path)!=record['engine_sha256'] or sha256(self.model/p['file'])!=record['onnx_sha256']
                    or record['onnx_sha256']!=p['sha256'] or record['GPU']!=self.cuda_info
                    or record['plan_sha256']!=self.index['plan_sha256'] or record['tensorrt']!=trt.__version__):
                    raise ValueError('Engine/ONNX/board mismatch in '+part)
                engine=self.trt_runtime.deserialize_cuda_engine(engine_path.read_bytes())
                if engine is None:raise RuntimeError('Cannot load '+part)
                self.engines.append(engine)
            self.assignments,slots=buffer_slots(self.plan)
            ptrs=[]
            for size in slots:
                ptr=self.cuda.malloc(size);self.devices.append(ptr);ptrs.append(ptr)
            values={key:ptrs[slot] for key,slot in self.assignments.items()}
            for name,key in self.plan['inputs'].items():self.shared[name]=(values[key],self.plan['buffers'][key]['shape'])
            for name,key in self.plan['outputs'].items():self.shared[name]=(values[key],self.plan['buffers'][key]['shape'])
            shape=[1,3]+manifest['input_size'];ptr=self.cuda.malloc(int(np.prod(shape))*4);self.devices.append(ptr)
            self.shared['images']=(ptr,shape)
            for name,(ptr,shape) in self.shared.items():
                host,array=self.cuda.host(shape);self.hosts.append(host);self.host_arrays[name]=array
            # All engines enqueue serially on the same stream. Their transient
            # activation/scratch memory can therefore use one shared allocation.
            size=max(e.device_memory_size for e in self.engines)
            scratch=self.cuda.malloc(max(size,256));self.devices.append(scratch)
            self.scratch_capsule=memory_capsule(scratch.value)
            for e in self.engines:
                context=e.create_execution_context_without_device_memory()
                if context is None:raise RuntimeError('Cannot create execution context')
                context.device_memory=self.scratch_capsule;self.contexts.append(context)
            encoder_bindings=[]
            expected={'images':[1,3]+manifest['input_size']}
            expected.update(dict(zip(manifest['feature_names'],manifest['feature_shapes'])))
            for i in range(encoder.num_bindings):
                name=encoder.get_binding_name(i)
                if (name not in expected or list(encoder.get_binding_shape(i))!=expected[name]
                    or encoder.get_binding_dtype(i)!=trt.float32 or encoder.get_binding_format(i)!=trt.TensorFormat.LINEAR):
                    raise ValueError('Encoder binding differs: '+name)
                encoder_bindings.append(self.shared[name][0].value)
            self.runners.append(encoder_bindings)
            dtypes={'float32':trt.float32,'float16':trt.float16,'int32':trt.int32,'bool':trt.bool}
            part_positions={n:i+1 for i,n in enumerate(self.plan['parts'])}
            self.calls=[]
            for step in self.plan['steps']:
                i=part_positions[step['part']];e=self.engines[i];p=self.plan['parts'][step['part']];bindings=[]
                if e.num_bindings!=len(p['inputs'])+len(p['outputs']):raise ValueError('Partition binding count differs')
                for j in range(e.num_bindings):
                    kind='inputs' if e.binding_is_input(j) else 'outputs';name=e.get_binding_name(j)
                    spec=p[kind][name]
                    if (list(e.get_binding_shape(j))!=spec['shape'] or e.get_binding_dtype(j)!=dtypes[spec['dtype']]
                        or e.get_binding_format(j)!=trt.TensorFormat.LINEAR or e.is_shape_binding(j)):
                        raise ValueError('Partition binding differs: '+step['part']+'/'+name)
                    bindings.append(values[step[kind][name]].value)
                self.calls.append((i,bindings))
            for _ in range(3):
                event=C.c_void_p();self.cuda.check(self.cuda.lib.cudaEventCreate(C.byref(event)),'partition event');self.events.append(event)
            self.record={'backend':'tensorrt','route':'hybrid','precision':'fp16',
                'execution':'GPU TensorRT FP16 encoder + physically partitioned GPU decoder; CPU preprocessing/postprocessing',
                'precision_scope':self.plan['policy'],'tensorrt':trt.__version__,'cuda':self.cuda_info,
                'engine_bindings':{'encoder':encoder_record,'decoder':{'policy':self.plan['policy'],
                    'partition_plan_sha256':self.index['plan_sha256'],'source_onnx_sha256':self.plan['source_decoder_sha256'],
                    'engine_count':len(self.plan['parts']),'parts':self.index['parts']}},
                'physical_engine_counts':self.plan['physical_engine_counts'],'decoder_dispatches':len(self.calls),
                'device_buffer_pool_bytes':sum(slots),'shared_context_scratch_bytes':size,
                'no_CPU_intermediate_transfers':True,'profile_enabled':False,'runtime':None}
        except BaseException:
            self.close();raise

    def infer(self,rgb,normalized):
        start=time.perf_counter();cuda=self.cuda
        np.copyto(self.host_arrays['images'],np.asarray(normalized,np.float32));src=self.host_arrays['images']
        cuda.check(cuda.lib.cudaMemcpyAsync(self.shared['images'][0],C.c_void_p(src.ctypes.data),src.nbytes,1,self.stream),'partition H2D')
        cuda.check(cuda.lib.cudaEventRecord(self.events[0],self.stream),'encoder begin')
        if not self.contexts[0].execute_async_v2(self.runners[0],self.stream.value):raise RuntimeError('Encoder execute failed')
        cuda.check(cuda.lib.cudaEventRecord(self.events[1],self.stream),'encoder end')
        for i,bindings in self.calls:
            if not self.contexts[i].execute_async_v2(bindings,self.stream.value):raise RuntimeError('Partition execute failed: '+str(i))
        cuda.check(cuda.lib.cudaEventRecord(self.events[2],self.stream),'decoder end')
        for name in self.manifest['output_names']+(self.manifest['feature_names'] if self.args.dump_raw else []):
            a=self.host_arrays[name]
            cuda.check(cuda.lib.cudaMemcpyAsync(C.c_void_p(a.ctypes.data),self.shared[name][0],a.nbytes,2,self.stream),'partition D2H')
        cuda.check(cuda.lib.cudaStreamSynchronize(self.stream),'partition synchronize')
        timing={}
        for key,a,b in (('gpu_encoder_ms',0,1),('gpu_decoder_ms',1,2),('gpu_compute_ms',0,2)):
            v=C.c_float();cuda.check(cuda.lib.cudaEventElapsedTime(C.byref(v),self.events[a],self.events[b]),key);timing[key]=float(v.value)
        timing['inference_ms']=(time.perf_counter()-start)*1000
        timing['transfer_submit_sync_overhead_ms']=max(0.,timing['inference_ms']-timing['gpu_compute_ms'])
        return ([self.host_arrays[n] for n in self.manifest['output_names']],timing,
            [self.host_arrays[n] for n in self.manifest['feature_names']] if self.args.dump_raw else [])

    def metadata(self):return self.record

    def close(self):
        if not hasattr(self,'cuda'):return
        if self.stream.value:self.cuda.lib.cudaStreamSynchronize(self.stream)
        self.contexts=[];self.engines=[]
        for event in self.events:self.cuda.lib.cudaEventDestroy(event)
        self.events=[]
        for ptr in self.devices:self.cuda.lib.cudaFree(ptr)
        self.devices=[]
        for ptr in self.hosts:self.cuda.lib.cudaFreeHost(ptr)
        self.hosts=[]
        if self.stream.value:self.cuda.lib.cudaStreamDestroy(self.stream);self.stream.value=None
        self.trt_runtime=None
