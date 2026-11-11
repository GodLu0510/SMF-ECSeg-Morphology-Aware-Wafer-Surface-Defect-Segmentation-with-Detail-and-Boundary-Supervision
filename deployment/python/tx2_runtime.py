"""Synchronous TensorRT 8.2 pipeline with CUDA 10.2 C API, Python 3.6 compatible.

Encoder features remain in device memory and feed the GPU decoder directly.
Pinned host buffers and explicit stream/event synchronization bound lifetimes.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from ecseg_common import sha256

class Cuda:
    def __init__(self):
        self.lib=C.CDLL('libcudart.so.10.2')
        signatures={
            'cudaGetDeviceCount':[C.POINTER(C.c_int)],'cudaSetDevice':[C.c_int],
            'cudaDeviceGetAttribute':[C.POINTER(C.c_int),C.c_int,C.c_int],
            'cudaRuntimeGetVersion':[C.POINTER(C.c_int)],'cudaDriverGetVersion':[C.POINTER(C.c_int)],
            'cudaMalloc':[C.POINTER(C.c_void_p),C.c_size_t],'cudaFree':[C.c_void_p],
            'cudaHostAlloc':[C.POINTER(C.c_void_p),C.c_size_t,C.c_uint],'cudaFreeHost':[C.c_void_p],
            'cudaMemcpyAsync':[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p],
            'cudaStreamCreate':[C.POINTER(C.c_void_p)],'cudaStreamDestroy':[C.c_void_p],
            'cudaStreamSynchronize':[C.c_void_p],'cudaEventCreate':[C.POINTER(C.c_void_p)],
            'cudaEventDestroy':[C.c_void_p],'cudaEventRecord':[C.c_void_p,C.c_void_p],
            'cudaEventElapsedTime':[C.POINTER(C.c_float),C.c_void_p,C.c_void_p]}
        for name,parameters in signatures.items():
            fn=getattr(self.lib,name);fn.argtypes=parameters;fn.restype=C.c_int
        self.lib.cudaGetErrorString.argtypes=[C.c_int];self.lib.cudaGetErrorString.restype=C.c_char_p
        self.check(self.lib.cudaSetDevice(0),'set device')
    def check(self,code,stage):
        if code:raise RuntimeError('CUDA %s: %s (%s)'%(stage,self.lib.cudaGetErrorString(code).decode(),code))
    def device_info(self):
        result={'device':0}
        serial=Path('/proc/device-tree/serial-number')
        try:result['board_serial_sha256']=hashlib.sha256(serial.read_bytes()).hexdigest()
        except OSError:result['board_serial_sha256']=None
        for key,attribute in (('compute_major',75),('compute_minor',76),('multiprocessors',16)):
            value=C.c_int();self.check(self.lib.cudaDeviceGetAttribute(C.byref(value),attribute,0),key);result[key]=value.value
        for name,key in (('cudaRuntimeGetVersion','runtime_version'),('cudaDriverGetVersion','driver_version')):
            value=C.c_int();self.check(getattr(self.lib,name)(C.byref(value)),key);result[key]=value.value
        return result
    def malloc(self,size):
        pointer=C.c_void_p();self.check(self.lib.cudaMalloc(C.byref(pointer),size),'malloc');return pointer
    def host(self,shape):
        size=int(np.prod(shape))*4
        pointer=C.c_void_p();self.check(self.lib.cudaHostAlloc(C.byref(pointer),size,0),'host alloc')
        array=np.ctypeslib.as_array(C.cast(pointer,C.POINTER(C.c_float)),shape=(size//4,)).reshape(shape)
        return pointer,array

class Engine:
    def __init__(self,args,manifest,out):
        import tensorrt as trt
        self.runtime=None # Shared benchmark's RKNN-only postprocessing hook stays disabled.
        self.cuda=Cuda();self.trt=trt
        self.stream=C.c_void_p();self.events=[];self.devices=[];self.hosts=[]
        self.engines=[];self.contexts=[];self.runners=[]
        self.args,self.manifest=args,manifest
        self.logger=trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger,'')
        self.trt_runtime=trt.Runtime(self.logger)
        self.cuda.check(self.cuda.lib.cudaStreamCreate(C.byref(self.stream)),'stream create')
        try:
            self.cuda_info=self.cuda.device_info()
            self.records={}
            for stage in ('encoder','decoder'):
                directory=Path(args.engines).resolve()/args.precision
                file=directory/(stage+'.engine')
                record=json.loads((directory/(stage+'.json')).read_text(encoding='utf-8'))
                onnx=Path(args.manifest).resolve().parent/(stage+'.onnx')
                if (record['engine_sha256']!=sha256(file) or record['onnx_sha256']!=sha256(onnx)
                    or record['manifest_sha256']!=sha256(args.manifest) or record['precision']!=args.precision
                    or record['tensorrt']!=trt.__version__ or record['GPU']!=self.cuda_info):
                    raise ValueError('Engine/model/runtime/GPU binding mismatch: '+stage)
                self.records[stage]=record
                engine=self.trt_runtime.deserialize_cuda_engine(file.read_bytes())
                if engine is None:raise RuntimeError('Cannot load '+str(file))
                self.engines.append(engine)
                ctx=engine.create_execution_context()
                if ctx is None:raise RuntimeError('Cannot create execution context')
                self.contexts.append(ctx)
            self._allocate()
            for _ in range(3):
                event=C.c_void_p();self.cuda.check(self.cuda.lib.cudaEventCreate(C.byref(event)),'event create');self.events.append(event)
        except BaseException:
            self.close();raise

    def _allocate(self):
        trt=self.trt
        expected=[({'images':tuple([1,3]+self.manifest['input_size'])},dict(zip(self.manifest['feature_names'],map(tuple,self.manifest['feature_shapes'])))),
                  (dict(zip(self.manifest['feature_names'],map(tuple,self.manifest['feature_shapes']))),dict(zip(self.manifest['output_names'],map(tuple,self.manifest['output_shapes']))))]
        shared={}
        for engine,(ins,outs) in zip(self.engines,expected):
            actual_in={engine.get_binding_name(i):tuple(engine.get_binding_shape(i)) for i in range(engine.num_bindings) if engine.binding_is_input(i)}
            actual_out={engine.get_binding_name(i):tuple(engine.get_binding_shape(i)) for i in range(engine.num_bindings) if not engine.binding_is_input(i)}
            if (ins,outs)!=(actual_in,actual_out):raise ValueError('Engine binding names/shapes differ')
            pointers=[]
            for i in range(engine.num_bindings):
                if engine.get_binding_dtype(i)!=trt.float32:raise ValueError('External binding is not FP32')
                if engine.get_binding_format(i)!=trt.TensorFormat.LINEAR:raise ValueError('External binding is not contiguous LINEAR')
                name=engine.get_binding_name(i);shape=tuple(engine.get_binding_shape(i))
                if name not in shared:
                    ptr=self.cuda.malloc(int(np.prod(shape))*4);self.devices.append(ptr);shared[name]=(ptr,shape)
                if shared[name][1]!=shape:raise ValueError('Shared feature shape differs')
                pointers.append(shared[name][0].value)
            self.runners.append(pointers)
        self.shared=shared
        self.host_arrays={}
        for name,(ptr,shape) in shared.items():
            host,array=self.cuda.host(shape);self.hosts.append(host);self.host_arrays[name]=array

    def infer(self,rgb,normalized):
        start=time.perf_counter();cuda=self.cuda
        np.copyto(self.host_arrays['images'],np.asarray(normalized,dtype=np.float32))
        source=self.host_arrays['images']
        cuda.check(cuda.lib.cudaMemcpyAsync(self.shared['images'][0],C.c_void_p(source.ctypes.data),source.nbytes,1,self.stream),'H2D')
        cuda.check(cuda.lib.cudaEventRecord(self.events[0],self.stream),'encoder begin')
        if not self.contexts[0].execute_async_v2(self.runners[0],self.stream.value):raise RuntimeError('Encoder execute failed')
        cuda.check(cuda.lib.cudaEventRecord(self.events[1],self.stream),'encoder end')
        if not self.contexts[1].execute_async_v2(self.runners[1],self.stream.value):raise RuntimeError('Decoder execute failed')
        cuda.check(cuda.lib.cudaEventRecord(self.events[2],self.stream),'decoder end')
        names=self.manifest['output_names']+(self.manifest['feature_names'] if self.args.dump_raw else [])
        # Only diagnostic runs copy intermediate features back to the CPU.
        for name in names:
            dest=self.host_arrays[name]
            cuda.check(cuda.lib.cudaMemcpyAsync(C.c_void_p(dest.ctypes.data),self.shared[name][0],dest.nbytes,2,self.stream),'D2H '+name)
        cuda.check(cuda.lib.cudaStreamSynchronize(self.stream),'stream synchronize')
        timings={}
        for key,a,b in (('gpu_encoder_ms',0,1),('gpu_decoder_ms',1,2),('gpu_compute_ms',0,2)):
            value=C.c_float();cuda.check(cuda.lib.cudaEventElapsedTime(C.byref(value),self.events[a],self.events[b]),key)
            timings[key]=float(value.value)
        # Outputs are valid until the next infer() call, not references to freed buffers.
        outputs=[self.host_arrays[n] for n in self.manifest['output_names']]
        features=[self.host_arrays[n] for n in self.manifest['feature_names']] if self.args.dump_raw else []
        timings['inference_ms']=(time.perf_counter()-start)*1000
        timings['transfer_submit_sync_overhead_ms']=max(0.,timings['inference_ms']-timings['gpu_compute_ms'])
        return outputs,timings,features

    def metadata(self):
        return {'backend':'tensorrt','route':'hybrid','execution':'GPU TensorRT encoder + GPU TensorRT decoder; CPU preprocessing/postprocessing',
            'precision':self.args.precision,'precision_scope':self.records['encoder']['policy'],
            'tensorrt':self.trt.__version__,'cuda':self.cuda_info,'engine_bindings':self.records,
            'intermediate_features':'P3/P4/P5 passed directly between GPU engines; D2H snapshots only when dump_raw>0',
            'cpu_threads':self.args.cpu_threads,'profile_enabled':False,'runtime':None,
            'timing_scope':'CUDA events measure GPU kernels; inference_ms includes pinned input copy, H2D/D2H, dispatch and synchronization'}

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
