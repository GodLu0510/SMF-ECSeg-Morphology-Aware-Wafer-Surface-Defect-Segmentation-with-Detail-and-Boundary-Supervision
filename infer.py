
"""Predict instance masks with a user-provided ECSeg-S/SMF-ECSeg checkpoint."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/smf_ecseg.yml'))
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--class-names', nargs='+', default=['Multi-ball', 'Scratch', 'Particle', 'Stain'])
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--no-ema', action='store_true')
    a = p.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    sys.path.insert(0, str(ROOT / 'ecdetseg'))
    sys.path.insert(0, str(ROOT / 'deployment/python'))
    from engine.core import YAMLConfig
    from ecseg_common import preprocess, decode
    from ecseg_benchmark import draw
    config, checkpoint, images, output = [Path(x).resolve() for x in
                                          (a.config, a.checkpoint, a.images, a.output)]
    if output.exists() and any(output.iterdir()):
        p.error('Choose a new or empty output directory.')
    files = [images] if images.is_file() else sorted(x for x in images.rglob('*') if x.suffix.lower() in
                                                    {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'})
    if not files:
        p.error('No input images found.')
    if images == output or images in output.parents or output in images.parents:
        p.error('Input and output directories must be separate.')
    device = torch.device(a.device)
    previous = Path.cwd()
    try:
        os.chdir(ROOT / 'ecdetseg')
        cfg = YAMLConfig(str(config)); cfg.yaml_cfg.setdefault('ECGraph', {})['skip_load_backbone'] = True
        if len(a.class_names) != cfg.yaml_cfg['num_classes']:
            p.error('--class-names must match num_classes and the trained class order.')
        model = cfg.model.to(device).eval()
        state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
        if not a.no_ema and isinstance(state.get('ema'), dict) and 'module' in state['ema']:
            weights = state['ema']['module']
        else:
            weights = state.get('model', state.get('state_dict', state))
        model.load_state_dict(weights, strict=True)
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream, torch.inference_mode():
            for index, path in enumerate(files):
                with Image.open(path) as opened:
                    image = opened.convert('RGB')
                data = torch.from_numpy(preprocess(image, cfg.yaml_cfg['eval_spatial_size'])[1]).to(device)
                prediction = model(data)
                tensors = [prediction[key].float().cpu().numpy() for key in
                           ('pred_logits', 'pred_boxes', 'pred_masks')]
                result = decode(tensors, image.size, a.class_names, 300, a.conf, 0.0)
                draw(image, result, output / ('%06d_' % index + path.stem + '.png'))
                stream.write(json.dumps({'image': path.name, 'status': 'ok', 'pass': 0,
                                         'detections': result}, ensure_ascii=False) + '\n')
    finally:
        os.chdir(previous)
    print('Predictions saved to:', output)

if __name__ == '__main__':
    main()
