#!/usr/bin/env python3
"""Export an actual ECSeg ECGraph checkpoint, keeping its trained A/B modules.

Run with the original EdgeCrafter PyTorch environment. Does not edit training
sources. Hybrid: backbone+encoder ONNX -> RKNN; decoder+mask ONNX -> CPU ORT.
Full is a compiler-dependent experimental route, never assumed all-NPU.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import os
from pathlib import Path
import sys

from ecseg_common import SCHEMA, MEAN, STD, OUTPUT_NAMES, dump_json, sha256, preprocess


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def export(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from torch import nn
    from PIL import Image
    source, checkpoint, config, out = [Path(p).resolve() for p in
                                        (args.source, args.checkpoint, args.config, args.output)]
    annotations = Path(args.annotations).resolve() if args.annotations else None
    verify_image = Path(args.verify_image).resolve()
    verify_images = list(dict.fromkeys([verify_image] + [Path(path).resolve() for path in getattr(args, "verify_images", [])]))
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if not (source / "engine/core").is_dir():
        raise ValueError("--source must be the original EdgeCrafter ecdetseg directory")
    for path in [checkpoint, config, annotations, *verify_images]:
        if path is None or not path.is_file():
            raise ValueError(f"Required export input does not exist: {path}")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty/new output directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    from engine.core import YAMLConfig
    torch.set_num_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(0)
    with working_directory(source):
        cfg = YAMLConfig(str(config))
        if cfg.yaml_cfg.get("model") != "ECGraph" or cfg.yaml_cfg.get("task") != "segmentation":
            raise ValueError("This exporter requires a segmentation ECGraph configuration")
        cfg.yaml_cfg.setdefault("ECGraph", {})["skip_load_backbone"] = True
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if args.model_weights:
            state, weights_source = ckpt["model"], "model"
        else:
            if not isinstance(ckpt.get("ema"), dict) or not isinstance(ckpt["ema"].get("module"), dict):
                raise ValueError("EMA weights are required; use --model-weights only when intentionally exporting model weights")
            state, weights_source = ckpt["ema"]["module"], "ema.module"
        print(f"Checkpoint={checkpoint}; weights={weights_source}; epoch={ckpt.get('last_epoch')}; strict=True", flush=True)
        model = cfg.model.cpu().eval()
        model.load_state_dict(state, strict=True)
        classes = cfg.yaml_cfg["num_classes"]
        size = list(cfg.yaml_cfg["eval_spatial_size"])
        if not cfg.yaml_cfg.get("use_focal_loss", True) or cfg.yaml_cfg.get("remap_mscoco_category", False):
            raise ValueError("Expected focal loss and remap_mscoco_category=false")
        if args.annotations:
            import json
            cats = json.loads(annotations.read_text(encoding="utf-8-sig"))["categories"]
            # Training's ConvertCocoPolysToMask uses category IDs directly if remap=false.
            cats = sorted(cats, key=lambda c: c["id"])
            if [c["id"] for c in cats] != list(range(classes)):
                raise ValueError("Training remap=false requires category IDs 0..num_classes-1")
            names = [c["name"] for c in cats]
        else:
            raise ValueError("--annotations is required to establish the real trained class order")
        with Image.open(verify_image) as opened:
            image = opened.convert("RGB")
        _, data = preprocess(image, size)
        data = torch.from_numpy(data)
        with torch.no_grad():
            original = model(data)
        model.deploy()

        class Encoder(nn.Module):
            def __init__(self, graph):
                super().__init__()
                self.graph = graph

            def forward(self, images):
                values = []
                for index, (sources, path, stage) in enumerate(self.graph._nodes):
                    get = lambda item: self.graph._input(item, index, values, images)
                    inputs = [get(item) for item in sources] if isinstance(sources, list) else get(sources)
                    if stage == "decoder":
                        return tuple(inputs)
                    values.append(self.graph.get_submodule(path)(inputs))

        class Decoder(nn.Module):
            def __init__(self, head):
                super().__init__()
                self.head = head

            def forward(self, p3, p4, p5):
                result = self.head([p3, p4, p5], spatial_feat=p3)
                return tuple(result[name] for name in OUTPUT_NAMES)

        class Full(nn.Module):
            def __init__(self, graph):
                super().__init__()
                self.graph = graph

            def forward(self, images):
                result = self.graph(images)
                return tuple(result[name] for name in OUTPUT_NAMES)

        encoder, decoder = Encoder(model).eval(), Decoder(model.decoder).eval()
        with torch.no_grad():
            features = encoder(data)
            deployed = decoder(*features)
        deploy_errors = {}
        for name, actual in zip(OUTPUT_NAMES, deployed):
            expected = original[name]
            deploy_errors[name] = float((actual - expected).abs().max())
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        feature_names = ["p3", "p4", "p5"]
        models = [("encoder.onnx", encoder, (data,), ["images"], feature_names, features),
                  ("decoder.onnx", decoder, tuple(features), feature_names, OUTPUT_NAMES, deployed)]
        if args.route == "full":
            models = [("full.onnx", Full(model).eval(), (data,), ["images"], OUTPUT_NAMES, deployed)]
        verification = {}
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        for filename, module, sample, inputs, outputs, expected in models:
            path = out / filename
            print(f"Exporting {path}", flush=True)
            with torch.no_grad():
                torch.onnx.export(module, sample, str(path), input_names=inputs, output_names=outputs,
                                  opset_version=17, do_constant_folding=True, dynamo=False)
            graph = onnx.load(str(path))
            # RKNN 2.3.0 lists Neg as unsupported. Sub(0,x) is algebraically
            # identical and keeps the RoPE computation and trained weights intact.
            # Infer the operand type instead of assuming every Neg is float32.
            inferred = onnx.shape_inference.infer_shapes(graph)
            types = {v.name: v.type.tensor_type.elem_type for v in
                     [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]}
            types.update({v.name: v.data_type for v in inferred.graph.initializer})
            for i, node in enumerate(graph.graph.node):
                if node.op_type == "Neg":
                    name = f"ecseg_neg_zero_{i}"
                    dtype = types.get(node.input[0])
                    if not dtype:
                        raise ValueError(f"Cannot determine dtype for {node.name}")
                    graph.graph.initializer.append(onnx.helper.make_tensor(name, dtype, [], [0]))
                    node.op_type = "Sub"
                    node.input.insert(0, name)
            onnx.save(graph, str(path))
            onnx.checker.check_model(graph)
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            actual = session.run(None, {name: value.numpy() for name, value in zip(inputs, sample)})
            errors = {}
            for name, a, b in zip(outputs, actual, expected):
                np.testing.assert_allclose(a, b.numpy(), rtol=3e-3, atol=3e-3)
                errors[name] = {"max_abs_error": float(np.max(np.abs(a-b.numpy()))),
                                "mean_abs_error": float(np.mean(np.abs(a-b.numpy())))}
            verification[filename] = {"sha256": sha256(path), "outputs": errors,
                                      "operators": dict(Counter(node.op_type for node in graph.graph.node))}
        # Also compare the complete split chain, including numerical errors in encoder outputs.
        if args.route == "hybrid":
            enc = ort.InferenceSession(str(out/"encoder.onnx"), options, providers=["CPUExecutionProvider"])
            dec = ort.InferenceSession(str(out/"decoder.onnx"), options, providers=["CPUExecutionProvider"])
            f = enc.run(None, {"images": data.numpy()})
            chain = dec.run(None, dict(zip(feature_names, f)))
            verification["split_chain"] = {}
            for name, actual in zip(OUTPUT_NAMES, chain):
                expected = original[name].numpy()
                np.testing.assert_allclose(actual, expected, rtol=6e-3, atol=6e-3)
                verification["split_chain"][name] = {"max_abs_error": float(np.max(np.abs(actual-expected))),
                                                     "mean_abs_error": float(np.mean(np.abs(actual-expected)))}
        additional_checks = []
        for extra_image in verify_images[1:]:
            with Image.open(extra_image) as opened:
                _, extra_input = preprocess(opened.convert("RGB"), size)
            with torch.no_grad():
                expected_outputs = model(torch.from_numpy(extra_input))
            if args.route == "hybrid":
                extra_features = enc.run(None, {"images": extra_input})
                extra_outputs = dec.run(OUTPUT_NAMES, dict(zip(feature_names, extra_features)))
            else:
                extra_outputs = session.run(OUTPUT_NAMES, {"images": extra_input})
            errors = {}
            for name, actual in zip(OUTPUT_NAMES, extra_outputs):
                expected = expected_outputs[name].numpy()
                np.testing.assert_allclose(actual, expected, rtol=6e-3, atol=6e-3)
                errors[name] = {"max_abs_error": float(np.max(np.abs(actual-expected))),
                                "mean_abs_error": float(np.mean(np.abs(actual-expected)))}
            additional_checks.append({"image": str(extra_image), "image_sha256": sha256(extra_image),
                                      "reference": "PyTorch deploy model", "outputs": errors})
            print(f"Verified additional image: {extra_image.name}", flush=True)
        manifest = {"schema": SCHEMA, "route": args.route, "batch_size": 1, "task": "instance_segmentation",
                    "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                    "checkpoint_epoch": ckpt.get("last_epoch"), "weights_source": weights_source,
                    "config": str(config), "config_sha256": sha256(config), "graph": model.graph,
                    "classes": names, "category_ids": list(range(classes)), "input_size": size,
                    "input_mean": MEAN, "input_std": STD, "resize": "PIL_bilinear_direct_no_letterbox",
                    "use_focal_loss": True, "num_top_queries": int(cfg.yaml_cfg.get("num_top_queries", 300)),
                    "feature_names": feature_names, "feature_shapes": [list(t.shape) for t in features],
                    "output_names": OUTPUT_NAMES, "output_shapes": [list(t.shape) for t in deployed],
                    "torch_version": torch.__version__, "opset": 17,
                    "onnx_files": {name: info["sha256"] for name, info in verification.items() if name.endswith(".onnx")}}
        dump_json(out/"manifest.json", manifest)
        dump_json(out/"verification.json", {"verify_image": str(verify_image),
                  "verify_image_sha256": sha256(verify_image), "deploy_max_abs_errors": deploy_errors,
                  "verification": verification, "additional_images": additional_checks,
                  "verified_image_count": len(verify_images),
                  "scope": "primary image: original PyTorch/deploy/ONNX; additional images: PyTorch deploy/ONNX; FP32 only, no RKNN/board validation"})
        (out/"classes.txt").write_text("\n".join(names)+"\n", encoding="utf-8")
        # Store the fully resolved configuration for reproducing this export.
        import yaml
        (out/"resolved_config.yaml").write_text(yaml.safe_dump(cfg.yaml_cfg, allow_unicode=True), encoding="utf-8")
        print(f"Verified export: {out}; classes={names}; outputs={manifest['output_shapes']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--verify-image", required=True)
    parser.add_argument("--verify-images", nargs="+", default=[], help="Additional real images for complete-chain verification")
    parser.add_argument("--output", required=True)
    parser.add_argument("--route", choices=["hybrid", "full"], default="hybrid")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model-weights", action="store_true", help="Use model instead of EMA weights")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
