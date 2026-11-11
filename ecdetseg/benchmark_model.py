"""Measure trained ECDet/ECSeg/ECGraph complexity, model size and PyTorch latency.

Run from ecdetseg/:
    python benchmark_model.py -c ../configs/yaml/ecseg_s_jingyuan_windows.yml \
        -r outputs/ecseg_s_jingyuan/best.pth --precision fp16

GFLOPs use PyTorch FlopCounterMode, including conv/matmul/attention. One MAC is
2 FLOPs; elementwise operations, normalization and resize are not counted.
FPS excludes file I/O, preprocessing, CPU/GPU input transfer and drawing.
"""
import argparse
from contextlib import nullcontext
import csv
import io
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from engine.core import YAMLConfig


SCRIPT_DIR = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Value must be a positive integer.')
    return number


def extract_weights(checkpoint, use_ema=True):
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a checkpoint dictionary or a tensor state_dict.')
    if use_ema and isinstance(checkpoint.get('ema'), dict) and 'module' in checkpoint['ema']:
        state, source = checkpoint['ema']['module'], 'ema.module'
    elif isinstance(checkpoint.get('model'), dict):
        state, source = checkpoint['model'], 'model'
    elif isinstance(checkpoint.get('state_dict'), dict):
        state, source = checkpoint['state_dict'], 'state_dict'
    elif checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        state, source = checkpoint, 'raw state_dict'
    else:
        raise ValueError('No usable model weights found in the checkpoint.')
    if state and all(key.startswith('module.') for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return state, source


def count_flops(model, height, width, device):
    # The MHA fused inference path bypasses the counted aten operators. Expand it
    # only during FLOP counting. Math SDPA exposes both QK^T and attention*V;
    # the original optimized kernels are restored before timing.
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        counter = FlopCounterMode(display=False)
        sample = torch.randn(1, 3, height, width, device=device)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), counter:
            outputs = model(sample)
        if not all(torch.isfinite(value).all().item() for value in outputs.values() if isinstance(value, torch.Tensor)):
            raise RuntimeError('Non-finite model outputs during FLOP counting.')
        operators = {str(op): int(value) for op, value in counter.get_flop_counts().get('Global', {}).items()}
        return int(counter.get_total_flops()), operators
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)


def autocast_context(device, precision):
    if precision == 'fp32':
        return nullcontext()
    dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def time_inference(forward, images, device, precision, warmup, iterations):
    timings = []
    with torch.inference_mode(), autocast_context(device, precision):
        for _ in range(warmup):
            outputs = forward(images)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        for _ in range(iterations):
            if device.type == 'cuda':
                start.record()
                outputs = forward(images)
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
            else:
                started = time.perf_counter()
                outputs = forward(images)
                timings.append((time.perf_counter() - started) * 1000)
        # A benchmark must not silently report fast but invalid FP16 results.
        tensors = outputs.values() if isinstance(outputs, dict) else [value for result in outputs for value in result.values()]
        for value in tensors:
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all().item():
                raise RuntimeError('Non-finite inference outputs. Try --precision fp32.')
    mean = statistics.fmean(timings)
    return {
        'fps': images.shape[0] * 1000 / mean,
        'latency_ms_mean': mean,
        'latency_ms_p50': float(np.percentile(timings, 50)),
        'latency_ms_p95': float(np.percentile(timings, 95)),
        'peak_cuda_memory_mb': torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else None,
    }


def run(args):
    # Resolve CLI files in the caller's directory, then use the training working
    # directory for relative graph and backbone paths inside YAML.
    config_path = Path(args.config).resolve(strict=True)
    checkpoint_path = Path(args.resume).resolve(strict=True)
    explicit_output = Path(args.output_dir).resolve() if args.output_dir else None
    os.chdir(SCRIPT_DIR)
    if args.warmup < 0:
        raise ValueError('--warmup must be non-negative.')
    device = torch.device(args.device)
    if device.type not in ('cuda', 'cpu'):
        raise ValueError('This benchmark supports CUDA and CPU.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment.')
    if device.type == 'cpu' and args.precision != 'fp32':
        raise ValueError('Use --precision fp32 on CPU.')
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('This device does not support BF16.')

    updates = {'eval_spatial_size': args.input_size} if args.input_size else {}
    cfg = YAMLConfig(str(config_path), **updates)
    for module_name in ('ViTAdapter', 'ECGraph'):
        if module_name in cfg.yaml_cfg:
            cfg.yaml_cfg[module_name]['skip_load_backbone'] = True
    print(f'Loading checkpoint: {checkpoint_path}', flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state, state_source = extract_weights(checkpoint, not args.no_ema)
    model = cfg.model
    model.load_state_dict(state, strict=True)
    original_params = sum(parameter.numel() for parameter in model.parameters())
    model = model.eval().deploy().float()
    deployed_params = sum(parameter.numel() for parameter in model.parameters())
    # Serialize only inference weights; exclude optimizer, EMA duplicate and logs.
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_bytes = buffer.tell()
    del buffer, checkpoint, state
    model.to(device)
    height, width = cfg.yaml_cfg['eval_spatial_size']
    print(f'Counting per-image FLOPs at {height}x{width} (including attention)...', flush=True)
    flops, operators = count_flops(model, height, width, device)
    images = torch.randn(args.batch_size, 3, height, width, device=device)
    print(f'Timing {args.iterations} iterations after {args.warmup} warmups, {args.precision}, batch={args.batch_size}...', flush=True)
    timing = time_inference(model, images, device, args.precision, args.warmup, args.iterations)
    report = {
        'config': str(config_path), 'checkpoint': str(checkpoint_path),
        'model_class': type(model).__name__, 'weight_source': state_source,
        'task': cfg.yaml_cfg['task'], 'num_classes': cfg.yaml_cfg['num_classes'],
        'input_size_hw': [height, width], 'batch_size': args.batch_size,
        'precision': args.precision, 'device': str(device),
        'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'cudnn_benchmark': torch.backends.cudnn.benchmark, 'tf32_enabled': False,
        'warmup_iterations': args.warmup, 'timed_iterations': args.iterations,
        'parameters_m': original_params / 1e6,
        'deployed_parameters_m': deployed_params / 1e6,
        'gflops': flops / 1e9, 'flop_counts_by_operator': operators,
        'model_size_mb': model_bytes / 1e6,
        'training_checkpoint_size_mb': checkpoint_path.stat().st_size / 1e6,
        'weights_dtype': 'fp32', 'model_deployed': True,
        'flops_method': 'PyTorch FlopCounterMode, per image, 1 MAC = 2 FLOPs; conv/matmul/attention; excludes elementwise/normalization/resize',
        'fps_scope': 'model forward only, including decoder mask head; excludes I/O, preprocessing, input transfer, postprocessing and drawing',
        'size_method': 'torch.save(deployed_model.state_dict()) in FP32; MB = 1,000,000 bytes',
        **timing,
    }
    if args.include_postprocess:
        postprocessor = cfg.postprocessor.to(device).eval()
        # Project targets use [width, height] as their orig_size convention.
        original_sizes = torch.tensor([[width, height]] * args.batch_size, device=device)
        print('Timing model + project postprocessing...', flush=True)
        def forward_with_postprocess(batch):
            return postprocessor(model(batch), original_sizes)
        report['with_postprocess'] = time_inference(forward_with_postprocess, images, device, args.precision, args.warmup, args.iterations)
        report['with_postprocess']['scope'] = 'model + detection/mask postprocessing; excludes I/O/preprocessing/input transfer/drawing'

    output = explicit_output or Path(cfg.yaml_cfg['output_dir']).resolve() / f'benchmark_{args.precision}_b{args.batch_size}'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    row = {key: report[key] for key in ('model_class', 'num_classes', 'hardware', 'precision', 'batch_size', 'gflops', 'parameters_m', 'deployed_parameters_m', 'fps', 'latency_ms_mean', 'model_size_mb', 'training_checkpoint_size_mb')}
    row.update(height=height, width=width)
    if args.include_postprocess:
        row['fps_with_postprocess'] = report['with_postprocess']['fps']
    with (output / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    summary = '\n'.join([
        f"Hardware:              {report['hardware']}",
        f"Input / batch:         {height}x{width} / {args.batch_size}",
        f"Inference precision:   {args.precision} (TF32 disabled)",
        f"GFLOPs / image:        {report['gflops']:.4f}",
        f"Parameters / M:        {report['parameters_m']:.4f} (before fusion)",
        f"Deploy parameters / M: {report['deployed_parameters_m']:.4f}",
        f"FPS (model only):      {report['fps']:.2f}",
        f"Latency / ms:          {report['latency_ms_mean']:.3f} (mean)",
        f"Model Size / MB:       {report['model_size_mb']:.4f} (deployed FP32 weights only)",
        f"Checkpoint Size / MB:  {report['training_checkpoint_size_mb']:.4f} (training checkpoint)",
        f"Weights selected:      {state_source}",
    ])
    if args.include_postprocess:
        summary += f"\nFPS (+ postprocess):    {report['with_postprocess']['fps']:.2f}"
    (output / 'metrics.txt').write_text(summary + '\n', encoding='utf-8')
    print('\n' + summary, flush=True)
    print(f'\nResults: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume', required=True, help='Trained checkpoint; default uses EMA weights when available.')
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    parser.add_argument('--batch-size', type=positive_int, default=1)
    parser.add_argument('--input-size', type=positive_int, nargs=2, metavar=('H', 'W'))
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', '--iters', type=positive_int, default=300)
    parser.add_argument('--include-postprocess', action='store_true')
    parser.add_argument('--no-ema', action='store_true', help='Use checkpoint model instead of EMA.')
    parser.add_argument('--output-dir')
    parser.add_argument('--seed', type=int, default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
