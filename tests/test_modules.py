
"""Behavior tests for real detail fusion, morphology gating and mask supervision."""
import sys
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ecdetseg'))
from engine.edgecrafter.modules.sdc import WaferDetailAdapter
from engine.edgecrafter.modules.mabg import WaferMorphologyBlock
from engine.edgecrafter.modules.fbas import sample_wafer_points, tolerant_boundary_loss
from engine.edgecrafter.ecvit import ViTAdapter
from engine.edgecrafter.hybrid_encoder import RepNCSPELAN4

class ModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sdc_preserves_semantic_path_with_zero_residual(self):
        kwargs = dict(name='ecseg_vitt', embed_dim=32, num_heads=4, depth=2,
                      interaction_indexes=[0, 1], skip_load_backbone=True)
        enhanced = WaferDetailAdapter(**kwargs).eval()
        baseline = ViTAdapter(**kwargs).eval()
        shared = {k: v for k, v in enhanced.state_dict().items() if not k.startswith('detail_')}
        baseline.load_state_dict(shared, strict=True)
        with torch.no_grad():
            enhanced.detail_scale.zero_()
            image = torch.randn(1, 3, 64, 64)
            actual, expected = enhanced(image), baseline(image)
        self.assertEqual([list(x.shape[-2:]) for x in actual], [[8, 8], [4, 4], [2, 2]])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_mabg_preserves_original_fusion_with_zero_residual(self):
        enhanced = WaferMorphologyBlock(32, 32, 64, 8, n=1).eval()
        baseline = RepNCSPELAN4(32, 32, 64, 8, n=1).eval()
        baseline.load_state_dict({k: v for k, v in enhanced.state_dict().items() if not k.startswith('morph_')}, strict=True)
        with torch.no_grad():
            enhanced.morph_scale.zero_()
            features = torch.randn(1, 32, 12, 12)
            torch.testing.assert_close(enhanced(features), baseline(features))

    def test_boundary_sampling_covers_tiny_foreground(self):
        target = torch.zeros(1, 1, 32, 32)
        target[:, :, 10, 14] = 1
        coordinates = sample_wafer_points(torch.full((1, 1, 8, 8), -20.), target, 1600)
        self.assertEqual(tuple(coordinates.shape), (1, 1600, 2))
        self.assertTrue(bool(((coordinates >= 0) & (coordinates <= 1)).all()))
        expected = torch.tensor([(14.5 / 32), (10.5 / 32)])
        torch.testing.assert_close(coordinates[0, 800:1200], expected.expand(400, 2))

    def test_empty_foreground_fallback_is_finite(self):
        coords = sample_wafer_points(torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 32, 32), 1600)
        self.assertTrue(bool(torch.isfinite(coords).all()))
        self.assertEqual(tuple(coords.shape), (2, 1600, 2))

    def test_boundary_loss_backpropagates_for_thin_target(self):
        target = torch.zeros(1, 1, 32, 32)
        target[:, :, 7:25, 14] = 1
        logits = torch.randn(1, 1, 8, 8, requires_grad=True)
        loss = tolerant_boundary_loss(logits, target, 1., tolerance=1)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_empty_ground_truth_boundary_has_zero_penalty(self):
        logits = torch.randn(1, 1, 8, 8, requires_grad=True)
        loss = tolerant_boundary_loss(logits, torch.zeros(1, 1, 32, 32), 1.)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

if __name__ == '__main__':
    unittest.main()
