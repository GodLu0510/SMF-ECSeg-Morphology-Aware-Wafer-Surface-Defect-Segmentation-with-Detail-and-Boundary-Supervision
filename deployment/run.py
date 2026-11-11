
"""Run a deployment stage using an explicit, initially blank paths configuration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    actions = ['export', 'convert-rknn', 'run-rknn', 'run-cpu', 'prepare-trt', 'build-trt',
               'run-trt', 'prepare-partition', 'build-partition', 'run-partition',
               'export-dynamic', 'run-onnx']
    p.add_argument('action', choices=actions)
    p.add_argument('--paths', required=True)
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    p.add_argument('--shape-mode', choices=['dynamic', 'static'], default='static')
    p.add_argument('--provider', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--output')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--save-images', action='store_true')
    p.add_argument('--print-command', action='store_true')
    a = p.parse_args()
    path_file = Path(a.paths).resolve()
    settings = json.loads(path_file.read_text(encoding='utf-8-sig'))
    def path(key, required=True):
        value = settings.get(key, '')
        if not value:
            if required:
                p.error('Fill the blank field "' + key + '" in ' + str(path_file))
            return None
        value = Path(value).expanduser()
        return str((value if value.is_absolute() else path_file.parent / value).resolve())
    def script(name):
        return [sys.executable, str(ROOT / 'python' / name)]
    def manifest():
        return path('manifest', False) or str(Path(path('export_dir')) / 'manifest.json')
    def data(command):
        command += ['--images', path('images'), '--warmup', str(a.warmup),
                    '--cpu-threads', str(a.cpu_threads)]
        annotation = path('annotations', False)
        if annotation:
            command += ['--coco', annotation]
        if a.output:
            command += ['--output', str(Path(a.output).resolve())]
        if a.save_images:
            command += ['--save-images']
        return command
    if a.action == 'export':
        command = script('ecseg_export.py') + ['--source', path('training_source'), '--config', path('training_config'),
                    '--checkpoint', path('checkpoint'), '--annotations', path('annotations'),
                    '--verify-image', path('verify_image'), '--output', path('export_dir'), '--route', 'hybrid']
    elif a.action == 'convert-rknn':
        command = script('ecseg_convert.py') + ['--manifest', manifest(), '--output', path('rknn_model'), '--precision', 'fp16']
    elif a.action in ('run-rknn', 'run-cpu'):
        command = script('ecseg_benchmark.py') + ['--manifest', manifest(), '--backend',
                    'rknn' if a.action == 'run-rknn' else 'onnx']
        if a.action == 'run-rknn':
            command += ['--model', path('rknn_model')]
        command = data(command)
    elif a.action == 'prepare-trt':
        command = script('prepare_trt.py') + ['--source', path('export_dir'), '--project', path('tensorrt_project')]
    elif a.action == 'build-trt':
        command = script('build_engines.py') + ['--project', path('tensorrt_project'), '--precision', a.precision,
                    '--stage', 'encoder' if a.precision == 'fp16' else 'both']
    elif a.action == 'run-trt':
        if a.precision != 'fp32':
            p.error('The final FP16 decoder route is run-partition. Use run-trt --precision fp32.')
        project = Path(path('tensorrt_project'))
        command = data(script('ecseg_tx2_benchmark.py') + ['--manifest', str(project / 'model/manifest.json'),
                    '--engines', str(project / 'engines'), '--precision', 'fp32'])
    elif a.action == 'prepare-partition':
        command = script('prepare_partitioned.py') + ['--project', path('tensorrt_project')]
    elif a.action == 'build-partition':
        command = script('build_partitioned.py') + ['--project', path('tensorrt_project')]
    elif a.action == 'run-partition':
        project = Path(path('tensorrt_project'))
        command = data(script('ecseg_partition_benchmark.py') + ['--manifest', str(project / 'model/manifest.json'),
                    '--engines', str(project / 'engines'), '--precision', 'fp16'])
    elif a.action == 'export-dynamic':
        command = script('export_dynamic_onnx.py') + ['--source', path('training_source'), '--config', path('training_config'),
                    '--checkpoint', path('checkpoint'), '--manifest', manifest(), '--verify-image', path('verify_image'),
                    '--output', path('dynamic_onnx_dir')]
    elif a.action == 'run-onnx':
        model_manifest = Path(path('dynamic_onnx_dir')) / a.shape_mode / 'manifest.json'
        command = data(script('ecseg_onnx_benchmark.py') + ['--manifest', str(model_manifest),
                    '--shape-mode', a.shape_mode, '--provider', a.provider])
    else:
        raise RuntimeError('Unknown deployment action')
    print(json.dumps(command, ensure_ascii=False, indent=2), flush=True)
    if not a.print_command:
        subprocess.run(command, cwd=str(ROOT), check=True)

if __name__ == '__main__':
    main()
