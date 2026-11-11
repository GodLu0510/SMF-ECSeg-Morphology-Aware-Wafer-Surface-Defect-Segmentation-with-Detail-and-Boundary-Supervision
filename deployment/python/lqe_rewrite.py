"""Equivalent stable TopK-before-normalization rewrite for the LQE branch."""
import copy
import numpy as np
import onnx
from onnx import helper, numpy_helper


def rewrite_lqe(model):
    """TopK(softmax(x)) = exp(TopK(x)-max(x)) / sum(exp(x-max(x)))."""
    model = copy.deepcopy(model)
    producers = {o: n for n in model.graph.node for o in n.output}
    replacements = {}
    removed = set()
    changes = []
    for top in model.graph.node:
        if top.op_type != 'TopK' or '/lqe_layers.' not in top.name:
            continue
        soft = producers[top.input[0]]
        assert soft.op_type == 'Softmax'
        attrs = {a.name: helper.get_attribute_value(a) for a in top.attribute}
        assert attrs.get('axis', -1) == -1 and attrs.get('largest', 1) == 1 and attrs.get('sorted', 1) == 1
        assert {a.name: helper.get_attribute_value(a) for a in soft.attribute}.get('axis', -1) == -1
        assert sum(soft.output[0] in n.input for n in model.graph.node) == 1
        base = top.name + '/TX2_FP32_LQE/'
        x = soft.input[0]
        axes = base + 'axes'
        model.graph.initializer.append(numpy_helper.from_array(np.array([-1], np.int64), axes))
        nodes = [
            helper.make_node('TopK', [x, top.input[1]], [base+'top_logits', top.output[1]], name=base+'TopLogits', **attrs),
            helper.make_node('ReduceMax', [x], [base+'maximum'], axes=[-1], keepdims=1, name=base+'Maximum'),
            helper.make_node('Sub', [x, base+'maximum'], [base+'shifted'], name=base+'CenterAll'),
            helper.make_node('Exp', [base+'shifted'], [base+'exponents'], name=base+'ExpAll'),
            helper.make_node('ReduceSum', [base+'exponents', axes], [base+'denominator'], keepdims=1, name=base+'Denominator'),
            helper.make_node('Sub', [base+'top_logits', base+'maximum'], [base+'top_shifted'], name=base+'CenterTop'),
            helper.make_node('Exp', [base+'top_shifted'], [base+'top_exponents'], name=base+'ExpTop'),
            helper.make_node('Div', [base+'top_exponents', base+'denominator'], [top.output[0]], name=base+'TopProbabilities'),
        ]
        replacements[top.name] = nodes
        removed.add(soft.name)
        changes.append({'original_softmax': soft.name, 'original_topk': top.name,
                        'input': x, 'output_probabilities': top.output[0], 'output_indices': top.output[1],
                        'replacement_nodes': [n.name for n in nodes]})
    assert len(changes) == 1, 'Expected exactly the final LQE branch in this static export'
    new_nodes = []
    for n in model.graph.node:
        if n.name in removed:
            continue
        new_nodes.extend(replacements.get(n.name, [n]))
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    original = {t.name: t.SerializeToString() for t in model.graph.initializer if '/TX2_FP32_LQE/' not in t.name}
    assert original
    onnx.checker.check_model(model)
    return model, changes
