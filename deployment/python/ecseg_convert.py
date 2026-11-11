#!/usr/bin/env python3
"""Model Zoo style ECSeg ONNX -> RKNN conversion, with verified model binding.

The RKNN config/load/build/export/release sequence follows Model Zoo 2.3.0's
MobileSAM encoder and YOLO11 examples. Dry-run and environment checks do not
import the SDK or PyTorch. No training source is needed for conversion.
"""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import traceback

from ecseg_common import dump_json, load_manifest, sha256


def prepare_calibration(images, out, size, limit):
    from PIL import Image
    root = Path(images).resolve()
    paths = sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        raise ValueError(f"No calibration images: {root}")
    if limit < 1:
        raise ValueError("--calibration-count must be positive")
    # Deterministic uniform coverage instead of just the first alphabetical files.
    count = min(limit, len(paths))
    indices = [round(i*(len(paths)-1)/max(count-1, 1)) for i in range(count)]
    directory = out / "calibration"
    directory.mkdir()
    records = []
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    for i, index in enumerate(indices):
        source = paths[index]
        path = directory / f"{i:06d}.png"
        with Image.open(source) as image:
            image.convert("RGB").resize((size[1], size[0]), bilinear).save(path)
        records.append({"source": str(source), "source_sha256": sha256(source), "resized": str(path)})
    dataset = out / "calibration.txt"
    # RKNN dataset format separates paths by whitespace.
    if any(any(c.isspace() for c in r["resized"]) for r in records):
        raise ValueError("Use a conversion output directory without spaces for calibration")
    dataset.write_text("\n".join(r["resized"] for r in records)+"\n", encoding="utf-8")
    dump_json(out/"calibration_sources.json", records)
    return dataset, count


def environment():
    packages = {}
    for name in ("rknn-toolkit2", "rknn-toolkit-lite2", "onnx", "numpy", "Pillow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    ready = platform.system() == "Linux" and packages["rknn-toolkit2"] is not None
    return {"system": platform.system(), "machine": platform.machine(), "python": sys.executable,
            "python_version": platform.python_version(), "packages": packages,
            "compiler_package_present": ready,
            "note": "Real build needs Linux and FULL rknn-toolkit2; Lite2 is for board inference, not model compilation"}


def plan_conversion(args):
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    filename = "encoder.onnx" if manifest["route"] == "hybrid" else "full.onnx"
    onnx_path = manifest_path.parent / filename
    if sha256(onnx_path) != manifest["onnx_files"][filename]:
        raise ValueError("ONNX hash differs from the verified export manifest")
    output = Path(args.output).resolve()
    if output.suffix.lower() != ".rknn" or output.exists() or output.with_suffix(".rknn.json").exists():
        raise ValueError("Choose a new .rknn output")
    if args.precision == "int8" and not args.calibration_images:
        raise ValueError("INT8 requires representative TRAINING images via --calibration-images")
    mean, std = manifest["input_mean"], manifest["input_std"]
    if (len(mean) != 3 or len(std) != 3 or any(not math.isfinite(v) for v in mean+std)
            or any(v <= 0 for v in std)):
        raise ValueError("Expected finite three-channel normalization with positive std")
    config = {"target_platform": "rk3588", "mean_values": [mean], "std_values": [std],
              "float_dtype": "float16", "optimization_level": getattr(args, "optimization_level", 3)}
    if args.precision == "int8":
        config.update(quantized_dtype="w8a8", quantized_algorithm=getattr(args, "algorithm", "normal"),
                      quantized_method="channel")
    if manifest["route"] == "hybrid":
        outputs, shapes = manifest["feature_names"], manifest["feature_shapes"]
    else:
        outputs, shapes = manifest["output_names"], manifest["output_shapes"]
    return {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "onnx": str(onnx_path), "onnx_sha256": sha256(onnx_path), "output": str(output),
            "route": manifest["route"], "precision": args.precision, "config": config,
            "input_name": "images", "input_shape": [1, 3, *manifest["input_size"]],
            "output_names": outputs, "output_shapes": shapes,
            "load_policy": "full_graph_verified_io",
            "build": {"do_quantization": args.precision == "int8"},
            "cpu_model": "decoder.onnx" if manifest["route"] == "hybrid" else None,
            "board_validation": "pending; conversion alone does not establish AP/FPS or all-NPU execution"}


def inspect_onnx(plan):
    import onnx
    from collections import Counter
    graph = onnx.load(plan["onnx"])
    onnx.checker.check_model(graph)
    def describe(values):
        return [{"name": value.name, "shape": [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in value.type.tensor_type.shape.dim], "type": value.type.tensor_type.elem_type}
                for value in values]
    inputs, outputs = describe(graph.graph.input), describe(graph.graph.output)
    if inputs != [{"name": plan["input_name"], "shape": plan["input_shape"], "type": onnx.TensorProto.FLOAT}]:
        raise ValueError(f"ONNX input differs from manifest: {inputs}")
    expected = [{"name": name, "shape": shape, "type": onnx.TensorProto.FLOAT}
                for name, shape in zip(plan["output_names"], plan["output_shapes"])]
    if outputs != expected:
        raise ValueError(f"ONNX output order/type/shape differs from manifest: {outputs}")
    return {"inputs": inputs, "outputs": outputs,
            "operators": dict(Counter(node.op_type for node in graph.graph.node))}


def convert(args):
    plan = plan_conversion(args)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if getattr(args, "dry_run", False):
        return plan
    if platform.system() != "Linux":
        raise RuntimeError("Actual RKNN compilation requires Linux + FULL rknn-toolkit2. "
                           "Copy examples/ecseg to your existing Linux conversion machine; --dry-run works on Windows.")
    try:
        from rknn.api import RKNN
        version = metadata.version("rknn-toolkit2")
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise RuntimeError("Activate the Linux Python with FULL rknn-toolkit2; Lite2 cannot compile models") from exc
    match = re.match(r"(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (2, 3):
        raise RuntimeError(f"Use RKNN-Toolkit2 >=2.3.0 for this converter; detected {version}")
    if not version.startswith("2.3.0"):
        print(f"Toolkit2={version}; verify compatibility with your board runtime/driver (original project uses 2.3.0)", flush=True)
    # Validate real graph metadata before asking the compiler to build it.
    graph_info = inspect_onnx(plan)
    output = Path(plan["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset, calibration_count = None, 0
    calibration_dir = output.parent / (output.stem + "_calibration")
    if args.precision == "int8":
        calibration_dir.mkdir(exist_ok=False)
        dataset, calibration_count = prepare_calibration(args.calibration_images, calibration_dir,
                                                        plan["input_shape"][2:], args.calibration_count)
    rknn = RKNN(verbose=True)
    temporary, stage = None, "config"
    def check(stage, ret):
        if ret not in (None, 0):
            raise RuntimeError(f"RKNN {stage} failed ({ret}); retain the compiler log for operator diagnosis")
    try:
        print(f"--> Config model (ImageNet normalization, {args.precision})", flush=True)
        check(stage, rknn.config(**plan["config"]))
        stage = "load_onnx"
        print("--> Loading encoder ONNX (preserve P3/P4/P5 outputs)", flush=True)
        # inspect_onnx already verifies the complete split graph's I/O. Use the
        # full-graph path validated with the static FP16 model on RK3588.
        check(stage, rknn.load_onnx(model=plan["onnx"]))
        stage = "build"
        print(f"--> Building RK3588 {args.precision}", flush=True)
        build = dict(plan["build"])
        if dataset:
            build["dataset"] = str(dataset)
        check(stage, rknn.build(**build))
        stage = "export_rknn"
        descriptor, name = tempfile.mkstemp(prefix=output.stem+".building_", suffix=".rknn", dir=str(output.parent))
        os.close(descriptor)
        temporary = Path(name).resolve()
        if temporary.parent != output.parent:
            raise RuntimeError("Unexpected temporary output directory")
        check(stage, rknn.export_rknn(str(temporary)))
        if temporary.stat().st_size == 0:
            raise RuntimeError("Compiler did not produce a nonempty RKNN file")
        record = {"manifest_sha256": plan["manifest_sha256"], "model_sha256": sha256(temporary),
                  "onnx_sha256": plan["onnx_sha256"], "target": "rk3588", "toolkit_version": version,
                  "precision": args.precision, "route": plan["route"],
                  "normalization": {"mean": plan["config"]["mean_values"][0], "std": plan["config"]["std_values"][0]},
                  "calibration_count": calibration_count, "calibration_sources": str(calibration_dir) if dataset else None,
                  "compiler_config": plan["config"], "onnx_graph": graph_info, "environment": environment(),
                  "load_policy": plan["load_policy"],
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "compiled; board inference and accuracy NOT yet verified"}
        if output.exists() or output.with_suffix(".rknn.json").exists():
            raise ValueError("Output appeared during compilation; refusing to overwrite it")
        stage = "publish_output"
        os.replace(str(temporary), str(output))
        dump_json(output.with_suffix(".rknn.json"), record)
        print(f"Created {output}\nBoard binding: {output.with_suffix('.rknn.json')}", flush=True)
        return record
    except BaseException as exc:
        dump_json(output.with_suffix(".rknn.failed.json"), {"stage": stage, "error": repr(exc),
                  "plan": plan, "toolkit_version": version, "traceback": traceback.format_exc()})
        raise
    finally:
        rknn.release()
        if temporary is not None and temporary.is_file():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--precision", choices=["fp16", "int8"], default="fp16",
                        help="fp16 means non-quantized RKNN build; compiler may retain some FP32 ops")
    parser.add_argument("--calibration-images")
    parser.add_argument("--calibration-count", type=int, default=200)
    parser.add_argument("--algorithm", choices=["normal", "mmse", "kl_divergence"], default="normal")
    parser.add_argument("--optimization-level", type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()
    if args.check_env:
        info = environment()
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return
    if not args.manifest or not args.output:
        parser.error("--manifest and --output are required for conversion")
    try:
        convert(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
