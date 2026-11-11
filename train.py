
"""Path-independent entry point for training and COCO evaluation."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/smf_ecseg.yml'))
    p.add_argument('--paths', default=str(ROOT / 'configs/paths.example.yml'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume')
    p.add_argument('--tuning')
    p.add_argument('--output-dir')
    p.add_argument('--test-only', action='store_true')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--use-amp', action='store_true', default=None)
    p.add_argument('--update', nargs='+', default=[])
    a = p.parse_args()
    config = Path(a.config).resolve()
    path_file = Path(a.paths).resolve()
    if not config.is_file() or not path_file.is_file():
        p.error('Config or paths file not found. See README.md.')
    paths = yaml.safe_load(path_file.read_text(encoding='utf-8-sig')) or {}
    def resolve(key, needed=False, directory=False):
        value = paths.get(key, '')
        if not value:
            if needed:
                p.error('Fill the blank field "' + key + '" in your paths file.')
            return None
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else path_file.parent / path).resolve()
        if not (path.is_dir() if directory else path.is_file()):
            p.error('Invalid path for ' + key + ': ' + str(path))
        return str(path)
    split = a.split if a.test_only else 'val'
    val_images = resolve(split + '_images', needed=True, directory=True)
    val_annotations = resolve(split + '_annotations', needed=True)
    updates = list(a.update)
    def override(key, value):
        # YAML string scalars avoid interpreting Windows paths as escapes.
        updates.append(key + '=' + repr(value))
    override('val_dataloader.dataset.img_folder', val_images)
    override('val_dataloader.dataset.ann_file', val_annotations)
    if not a.test_only:
        override('train_dataloader.dataset.img_folder', resolve('train_images', needed=True, directory=True))
        override('train_dataloader.dataset.ann_file', resolve('train_annotations', needed=True))
    resume = str(Path(a.resume).resolve()) if a.resume else resolve('resume_checkpoint')
    tuning = str(Path(a.tuning).resolve()) if a.tuning else resolve('pretrained_checkpoint')
    if a.test_only and not resume:
        resume = resolve('inference_checkpoint', needed=True)
    if resume and tuning:
        tuning = None  # Resume already contains complete model/training state.
    output = str(Path(a.output_dir).resolve()) if a.output_dir else str(ROOT / 'outputs' / config.stem)
    sys.path.insert(0, str(ROOT / 'ecdetseg'))
    import runpy
    native = runpy.run_path(str(ROOT / 'ecdetseg/train.py'), run_name='smf_training_entry')
    args = argparse.Namespace(config=str(config), resume=resume, tuning=tuning, device=a.device,
                              seed=a.seed, use_amp=a.use_amp, output_dir=output, summary_dir=None,
                              test_only=a.test_only, update=updates, print_method='builtin',
                              print_rank=0, local_rank=None)
    previous = Path.cwd()
    try:
        os.chdir(ROOT / 'ecdetseg')
        native['main'](args)
    finally:
        os.chdir(previous)

if __name__ == '__main__':
    main()
