
"""Model-independent tests of exported mask semantics and deployment entry points."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deployment/python'))
from ecseg_common import decode, preprocess
from ecseg_evaluation import evaluate_prediction_stream

class DeploymentTests(unittest.TestCase):
    def test_preprocessing_matches_imagenet_rgb_normalization(self):
        image = Image.fromarray(np.full((8, 12, 3), [0, 128, 255], dtype=np.uint8))
        rgb, value = preprocess(image, [16, 16])
        self.assertEqual(rgb.shape, (1, 16, 16, 3))
        self.assertEqual(value.shape, (1, 3, 16, 16))
        expected = (np.array([0, 128, 255]) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
        np.testing.assert_allclose(value[0, :, 0, 0], expected, rtol=1e-6)

    def test_masks_restore_to_original_image_dimensions(self):
        from pycocotools import mask as mask_util
        result = decode([np.array([[[10.]]], np.float32), np.array([[[.5, .5, 1., 1.]]], np.float32),
                         np.ones((1, 1, 4, 4), np.float32)], (40, 24), ['Particle'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['segmentation']['size'], [24, 40])
        self.assertEqual(result[0]['mask_area_pixels'], 960)
        self.assertEqual(result[0]['bbox_xyxy'], [0., 0., 40., 24.])

    def test_nonfinite_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            decode([np.array([[[np.nan]]]), np.ones((1, 1, 4)), np.ones((1, 1, 4, 4))], (8, 8), ['Particle'])

    def test_blank_paths_fail_before_running_export(self):
        result = subprocess.run([sys.executable, str(ROOT / 'deployment/run.py'), 'export', '--paths',
                                 str(ROOT / 'deployment/paths.example.json')], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Fill the blank field', result.stderr)

    def test_coherent_perfect_mask_prediction_has_unit_ap(self):
        from pycocotools import mask as mask_util
        mask = np.zeros((32, 32), np.uint8); mask[8:16, 8:16] = 1
        rle = mask_util.encode(np.asfortranarray(mask)); rle['counts'] = rle['counts'].decode('ascii')
        dataset = {'images': [{'id': 1, 'file_name': 'synthetic.png', 'width': 32, 'height': 32}],
                   'categories': [{'id': 0, 'name': 'Particle'}],
                   'annotations': [{'id': 1, 'image_id': 1, 'category_id': 0, 'bbox': [8, 8, 8, 8],
                                    'area': 64, 'iscrowd': 0, 'segmentation': rle}]}
        row = {'image': 'synthetic.png', 'pass': 0, 'status': 'ok', 'detections': [
            {'class_id': 0, 'score': .99, 'bbox_xyxy': [8, 8, 16, 16], 'segmentation': rle}]}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); predictions = folder / 'predictions.jsonl'
            predictions.write_text(json.dumps(row) + '\n', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = evaluate_prediction_stream(dataset, predictions, ['Particle'], [0], folder / 'evaluation')
            self.assertAlmostEqual(metrics['segm']['map50_95'], 1.)
            self.assertAlmostEqual(metrics['bbox']['map50_95'], 1.)
            segm = json.loads((folder / 'evaluation/predictions.segm.coco.json').read_text())
            self.assertNotIn('bbox', segm[0])

if __name__ == '__main__':
    unittest.main()
