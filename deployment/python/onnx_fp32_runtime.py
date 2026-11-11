"""FP32 ONNX Runtime CUDA/CPU, real dynamic H/W and explicit EP assignment audit."""
import ctypes as C
import gc
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from ecseg_common import sha256,preprocess


def read_profile_events(path):
    """Read ORT profiles, repairing only dangling array commas outside strings.

    ORT 1.11 emits [...,] for some optional kernel inputs (e.g. Clip).
    The original is evidence: retain it, save a normalized sidecar, and report
    every removed comma. No events/fields are dropped and other errors fail.
    """
    path=Path(path);raw=path.read_text(encoding='utf-8')
    info={'path':str(path),'sha256':sha256(path),'parse_mode':'strict',
          'removed_array_comma_offsets':[]}
    try:
        events=json.loads(raw)
    except json.JSONDecodeError as error:
        quoted=False;escaped=False;offsets=[]
        for index,char in enumerate(raw):
            if quoted:
                if escaped:escaped=False
                elif char=='\\':escaped=True
                elif char=='"':quoted=False
            elif char=='"':quoted=True
            elif char==',':
                following=index+1
                while following<len(raw) and raw[following] in ' \t\r\n':following+=1
                if following<len(raw) and raw[following]==']':offsets.append(index)
        if not offsets:raise
        removed=set(offsets)
        normalized=''.join(char for index,char in enumerate(raw) if index not in removed)
        # Strictly reparse the entire document before publishing any result.
        events=json.loads(normalized)
        sidecar=path.with_name(path.stem+'.normalized.json')
        sidecar.write_text(normalized,encoding='utf-8')
        info.update(parse_mode='repaired_trailing_array_commas',
            original_error=str(error),removed_array_comma_offsets=offsets,
            normalized_path=str(sidecar),normalized_sha256=sha256(sidecar))
    if not isinstance(events,list) or any(not isinstance(event,dict) for event in events):
        raise ValueError('ORT profile must be an array of event objects: '+str(path))
    info['event_count']=len(events)
    return events,info


def summarize_profiles(paths):
    providers={};nodes={};sources={}
    for label,path in paths.items():
        events,sources[label]=read_profile_events(path)
        for event in events:
            args=event.get('args',{});provider=args.get('provider')
            if event.get('cat')!='Node' or not provider:continue
            key=(label,provider,event.get('name',''),args.get('op_name','unknown'))
            row=nodes.setdefault(key,{'session':label,'provider':provider,'node':key[2],'op':key[3],
                'events':0,'duration_us':0.})
            row['events']+=1;row['duration_us']+=float(event.get('dur',0))
            p=providers.setdefault(provider,{'events':0,'duration_us':0.})
            p['events']+=1;p['duration_us']+=float(event.get('dur',0))
    return {'scope':'Separate representative initialization inference, excluded from timed benchmark; not a profile of every measured image',
        'CPU_assignment_note':'CPU nodes include intentional shape/control placement, not only unsupported CUDA math kernels',
        'profile_sources':sources,
        'providers':providers,'nodes':list(nodes.values()),
        'CPU_fallback_nodes':[r for r in nodes.values() if r['provider']=='CPUExecutionProvider'],
        'CUDA_nodes':[r for r in nodes.values() if r['provider']=='CUDAExecutionProvider']}


class Engine:
    def __init__(self,args,manifest,out):
        import onnxruntime as ort
        self.runtime=None;self.ort=ort;self.args=args;self.manifest=manifest;self.out=Path(out)
        self.sessions=[];self.encoder=None;self.decoder=None;self.profiles={};self.audit=None;self.cuda=None
        self.provider=args.provider;self.size=manifest['input_size'];self.base=Path(args.manifest).resolve().parent
        self.labels=('full',) if manifest['route']=='full' else ('encoder','decoder')
        if args.precision!='fp32' or manifest['precision']!='FP32':raise ValueError('This route requires FP32')
        if args.shape_mode!=manifest['shape_mode']:raise ValueError('Requested static/dynamic mode differs from actual model')
        self.available=ort.get_available_providers()
        if self.provider=='cuda':
            if 'CUDAExecutionProvider' not in self.available:
                raise RuntimeError('CUDAExecutionProvider is unavailable; refusing to relabel CPU inference as GPU: '+str(self.available))
            from tx2_runtime import Cuda
            self.cuda=Cuda();self.cuda.lib.cudaDeviceSynchronize.argtypes=[];self.cuda.lib.cudaDeviceSynchronize.restype=C.c_int
        try:
            # EP assignment is collected in separate sessions. The measured
            # sessions below have profiling OFF unless explicitly requested.
            self._sessions(True)
            from evaluation import collect_images
            candidates = collect_images(Path(args.images))
            image = candidates[0] if candidates else None
            if image is None:
                raise ValueError('No representative image found in the user-supplied --images directory')
            with Image.open(image) as im:rgb,normalized=preprocess(im,self.size)
            self.infer(rgb,normalized)
            paths=self._close_sessions()
            self.audit=summarize_profiles(paths)
            for label,source in self.audit['profile_sources'].items():
                if source['parse_mode']!='strict':
                    print('ORT profile JSON normalized: session='+label+', dangling array commas='+
                        str(len(source['removed_array_comma_offsets']))+'; original retained: '+source['path'],flush=True)
            if self.provider=='cuda' and not any(not r['op'].startswith('Memcpy') for r in self.audit['CUDA_nodes']):
                raise RuntimeError('CUDA requested but representative profile contains no CUDA nodes')
            (self.out/'provider_assignments.json').write_text(json.dumps(self.audit,indent=2)+'\n',encoding='utf-8')
            print('ONNX provider audit (outside measurement): '+json.dumps(self.audit['providers']),flush=True)
            gc.collect();self._sync();self._sessions(bool(args.profile))
        except BaseException:
            self.close();raise

    def _sync(self):
        if self.cuda:self.cuda.check(self.cuda.lib.cudaDeviceSynchronize(),'ORT device synchronize')

    def _sessions(self,profile):
        ort=self.ort
        if self.provider=='cuda':
            options={'device_id':0,'cudnn_conv_algo_search':'HEURISTIC','do_copy_in_default_stream':1,
                'gpu_mem_limit':int(self.args.gpu_mem_limit_mb)*1024*1024}
            requested=[('CUDAExecutionProvider',options),'CPUExecutionProvider']
        else:requested=['CPUExecutionProvider']
        self.sessions=[]
        for name in self.labels:
            filename=name+'.onnx';path=self.base/filename
            if sha256(path)!=self.manifest['onnx_files'][filename]:raise ValueError('ONNX checksum mismatch: '+filename)
            opt=ort.SessionOptions();opt.intra_op_num_threads=self.args.cpu_threads;opt.inter_op_num_threads=1
            opt.enable_profiling=profile;opt.profile_file_prefix=str(self.out/('ort_'+name+('_audit' if self.audit is None else '_measured')))
            opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session=ort.InferenceSession(str(path),opt,providers=requested)
            if self.provider=='cuda' and 'CUDAExecutionProvider' not in session.get_providers():
                raise RuntimeError('CUDA session initialization failed; actual providers='+str(session.get_providers()))
            if hasattr(session,'disable_fallback'):session.disable_fallback()
            if any(v.type!='tensor(float)' for v in session.get_inputs()+session.get_outputs()):
                raise ValueError('Expected FLOAT32 graph interface')
            for value in session.get_inputs():
                if value.shape[0]!=1:raise ValueError('This deployment fixes batch at 1')
                if self.args.shape_mode=='static' and any(not isinstance(d,int) for d in value.shape):
                    raise ValueError('Static model contains symbolic input dimensions')
                if self.args.shape_mode=='static' and value.name=='images' and value.shape[2:]!=self.size:
                    raise ValueError('Static graph input size differs from requested size')
                if self.args.shape_mode=='dynamic' and not any(isinstance(d,str) or d is None for d in value.shape[2:]):
                    raise ValueError('Dynamic-spatial model has fixed input H/W')
            self.sessions.append(session)
        self.encoder=self.sessions[0];self.decoder=self.sessions[1] if len(self.sessions)>1 else None

    def infer(self,rgb,normalized):
        start=time.perf_counter();h,w=self.size
        if normalized.shape!=(1,3,h,w):raise ValueError('Unexpected actual input shape')
        x=np.ascontiguousarray(normalized,dtype=np.float32)
        if self.manifest['route']=='full':
            if self.provider=='cuda':
                binding=self.encoder.io_binding();binding.bind_cpu_input('images',x)
                for name in self.manifest['output_names']:binding.bind_output(name,'cuda',0)
                self.encoder.run_with_iobinding(binding);self._sync();compute_end=time.perf_counter()
                values=binding.copy_outputs_to_cpu();self._sync();end=time.perf_counter()
            else:
                values=self.encoder.run(self.manifest['output_names'],{'images':x});compute_end=end=time.perf_counter()
            if any(v.dtype!=np.float32 for v in values):raise ValueError('Non-FP32 output')
            if values[2].shape[-2:]!=(h//4,w//4):raise ValueError('Mask dimensions do not follow actual H/W')
            return values,{'onnx_full_ms':(compute_end-start)*1000,'ort_output_copy_ms':(end-compute_end)*1000,
                'inference_ms':(end-start)*1000},[]
        if self.provider=='cuda':
            enc=self.encoder.io_binding();enc.bind_cpu_input('images',x)
            for name in self.manifest['feature_names']:enc.bind_output(name,'cuda',0)
            self.encoder.run_with_iobinding(enc);self._sync();enc_end=time.perf_counter()
            features=enc.get_outputs();dec=self.decoder.io_binding()
            if any(v.device_name().lower()!='cuda' for v in features):raise RuntimeError('Encoder features are not CUDA OrtValues')
            for name,value in zip(self.manifest['feature_names'],features):dec.bind_ortvalue_input(name,value)
            for name in self.manifest['output_names']:dec.bind_output(name,'cuda',0)
            self.decoder.run_with_iobinding(dec);self._sync();dec_end=time.perf_counter()
            values=dec.copy_outputs_to_cpu();self._sync();copy_end=time.perf_counter()
            saved_features=[v.numpy() for v in features] if self.args.dump_raw else []
            self._sync()
        else:
            features=self.encoder.run(self.manifest['feature_names'],{'images':x});enc_end=time.perf_counter()
            values=self.decoder.run(self.manifest['output_names'],dict(zip(self.manifest['feature_names'],features)))
            dec_end=copy_end=time.perf_counter();saved_features=features if self.args.dump_raw else []
        timing={'onnx_encoder_ms':(enc_end-start)*1000,'onnx_decoder_ms':(dec_end-enc_end)*1000,
            'ort_output_copy_ms':(copy_end-dec_end)*1000,'inference_ms':(time.perf_counter()-start)*1000}
        if any(v.dtype!=np.float32 for v in values):raise ValueError('Non-FP32 output')
        if values[2].shape[-2:]!=(h//4,w//4):raise ValueError('Mask spatial dimensions do not follow actual H/W')
        return values,timing,saved_features

    def metadata(self):
        return {'backend':'onnxruntime','route':self.manifest['route'],'precision':'fp32','shape_mode':self.args.shape_mode,
            'execution':'ONNX Runtime CUDA (explicit CPU fallback recorded)' if self.provider=='cuda' else 'ONNX Runtime CPU',
            'precision_scope':'FLOAT32 encoder and decoder; no TensorRT EP, FP16 or INT8 conversion',
            'onnxruntime_version':self.ort.__version__,'available_providers':self.available,
            'actual_session_providers':{n:s.get_providers() for n,s in zip(self.labels,self.sessions)},
            'provider_options':{n:s.get_provider_options() for n,s in zip(self.labels,self.sessions)},
            'graph_input_shapes':{n:{v.name:v.shape for v in s.get_inputs()} for n,s in zip(self.labels,self.sessions)},
            'graph_output_shapes':{n:{v.name:v.shape for v in s.get_outputs()} for n,s in zip(self.labels,self.sessions)},
            'provider_assignment_audit':self.audit,'provider_assignment_file':'provider_assignments.json',
            'input_shape':[1,3]+list(self.size),'dynamic_spatial':self.args.shape_mode=='dynamic','dynamic_batch':False,
            'intermediate_features':'Internal to one full ONNX graph' if self.manifest['route']=='full' else 'Explicit session feature transfer',
            'profile_enabled':bool(self.args.profile),'cpu_threads':self.args.cpu_threads,
            'cuda':self.cuda.device_info() if self.cuda else None,'runtime':None}

    def _close_sessions(self):
        paths={}
        for name,session in zip(self.labels,self.sessions):
            if session.get_session_options().enable_profiling:paths[name]=session.end_profiling()
        self.sessions=[];self.encoder=None;self.decoder=None
        return paths

    def close(self):
        self._sync()
        if self.sessions:self._close_sessions()
