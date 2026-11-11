"""Optional host-side numerical checks for the actual TX2 graph rewrites."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deployment/python'))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('onnx', 'onnxruntime'))


@unittest.skipUnless(AVAILABLE, 'Install deployment/requirements_export.txt for ONNX checks')
class OnnxAdaptationTests(unittest.TestCase):
    def compare(self, operation, input_shapes, output_shape, initializers=(), attributes=None):
        import onnx
        import onnxruntime as ort
        from onnx import helper as h, numpy_helper as n, TensorProto as t
        from prepare_trt import adapt
        names = list(input_shapes) + [name for name, _ in initializers]
        graph = h.make_graph(
            [h.make_node(operation, names, ['output'], name='synthetic_' + operation, **(attributes or {}))],
            'synthetic',
            [h.make_tensor_value_info(name, t.FLOAT, shape) for name, shape in input_shapes.items()],
            [h.make_tensor_value_info('output', t.FLOAT, output_shape)],
            [n.from_array(value, name) for name, value in initializers])
        model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=7)
        converted, counts = adapt(model)
        self.assertEqual(counts[operation], 1)
        self.assertFalse(any(node.op_type == operation for node in converted.graph.node))
        rng = np.random.default_rng(2026)
        inputs = {name: rng.uniform(-2, 2, size=shape).astype(np.float32)
                  for name, shape in input_shapes.items()}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        values = []
        for item in (model, converted):
            onnx.checker.check_model(item)
            session = ort.InferenceSession(item.SerializeToString(), options,
                                           providers=['CPUExecutionProvider'])
            values.append(session.run(None, inputs)[0])
        np.testing.assert_allclose(values[0], values[1], rtol=1e-5, atol=1e-5)

    def test_layer_normalization_rewrite(self):
        self.compare('LayerNormalization', {'image': [2, 3, 4]}, [2, 3, 4],
                     [('scale', np.array([.5, 1., 1.5, 2.], np.float32)),
                      ('bias', np.array([-.1, 0., .1, .2], np.float32))],
                     {'axis': -1, 'epsilon': 1e-5})

    def test_grid_sample_rewrite_including_outside_image(self):
        self.compare('GridSample', {'image': [2, 3, 5, 7], 'grid': [2, 4, 6, 2]}, [2, 3, 4, 6],
                     attributes={'align_corners': 0, 'mode': 'bilinear', 'padding_mode': 'zeros'})

    def test_instance_mask_projection_rewrite(self):
        self.compare('Einsum', {'features': [1, 4, 3, 5], 'queries': [1, 300, 4]}, [1, 300, 3, 5],
                     attributes={'equation': 'bchw,bnc->bnhw'})


if __name__ == '__main__':
    unittest.main()
