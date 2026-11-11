#!/usr/bin/env python3
"""ECSeg TX2 TensorRT GPU inference, COCO AP and board telemetry."""
import argparse
import contextlib
import csv
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from PIL import Image

from benchmark import percentiles, write_csv
from ecseg_common import decode, dump_json, load_manifest, preprocess, sha256
from ecseg_evaluation import evaluate_prediction_stream
from tx2_telemetry import Sampler, board_snapshot
from evaluation import collect_images, load_dataset, path_within

ROOT = Path(__file__).resolve().parents[1]


from tx2_runtime import Engine


def evaluate_instances(dataset, predictions, names, mapping, out, max_dets):
    """Legacy in-memory evaluator; production runs use streamed per-image evaluation."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = dataset
    gt.createIndex()
    results = {}
    log = io.StringIO()
    for kind in ("bbox", "segm"):
        # Exclude bbox from segmentation results so loadRes uses mask area for
        # small/medium/large segmentation AP, as COCO requires.
        items = [{key: value for key, value in p.items() if key != ("bbox" if kind == "segm" else "segmentation")}
                 for p in predictions]
        dump_json(out/f"predictions.{kind}.coco.json", items)
        with contextlib.redirect_stdout(log):
            if items:
                detected = gt.loadRes(items)
            else:
                detected = COCO()
                detected.dataset = {"images": dataset["images"], "categories": dataset["categories"], "annotations": []}
                detected.createIndex()
            evaluator = COCOeval(gt, detected, kind)
            evaluator.params.maxDets = [1, 10, max_dets]
            evaluator.params.imgIds = [im["id"] for im in dataset["images"]]
            evaluator.params.catIds = list(mapping)
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            precision = evaluator.eval["precision"]
            def average(values):
                valid = values[values > -1]
                return float(valid.mean()) if valid.size else None
            per_class = []
            for label, category in enumerate(mapping):
                index = list(evaluator.params.catIds).index(category)
                p = precision[:, :, index, 0, -1]
                per_class.append({"name": names[label], "category_id": category,
                                  "map50_95": average(p), "map50": average(p[0]),
                                  "ground_truth": sum(a["category_id"] == category and not a.get("iscrowd", 0)
                                                      for a in dataset["annotations"])})
            results[kind] = {"map50_95": average(precision[:, :, :, 0, -1]),
                             "map50": average(precision[0, :, :, 0, -1]),
                             "per_class": per_class, "max_dets": max_dets,
                             "coco_stats": [float(v) if v >= 0 else None for v in evaluator.stats]}
    (out/"coco_evaluation.log").write_text(log.getvalue(), encoding="utf-8")
    write_csv(out/"per_class.csv", [{"task": kind, **row} for kind, value in results.items() for row in value["per_class"]])
    return results


def draw(image, detections, output):
    import cv2
    from pycocotools import mask as mask_util
    rgb = np.asarray(image.convert("RGB")).copy()
    for item in reversed(detections):
        color = np.array([(37*item["class_id"]+60)%255, (97*item["class_id"]+160)%255,
                          (151*item["class_id"]+80)%255], dtype=np.uint8)
        rle = dict(item["segmentation"])
        rle["counts"] = rle["counts"].encode("ascii")
        mask = mask_util.decode(rle).astype(bool)
        rgb[mask] = (0.55*rgb[mask]+0.45*color).astype(np.uint8)
        box = [int(v) for v in item["bbox_xyxy"]]
        cv2.rectangle(rgb, tuple(box[:2]), tuple(box[2:]), color.tolist(), 2)
        cv2.putText(rgb, f"{item['class_name']} {item['score']:.2f}", (box[0], max(15, box[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1, cv2.LINE_AA)
    Image.fromarray(rgb).save(output)


def report(out, summary):
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"
    perf = summary["performance"]
    lines = ["# ECSeg / Jetson TX2 GPU运行报告", "", f"- 执行方式：{summary['engine'].get('execution', 'initialization failed')}",
             f"- 状态：{summary['status']}；精度状态：{summary['accuracy_status']}",
             f"- 成功/失败帧：{perf['frames_success']} / {perf['frames_failed']}；完成轮数：{perf['completed_passes']}",
             f"- 测量时间：{fmt(perf['measurement_duration_s'])} s；实际 batch=1",
             f"- 推理 FPS：{fmt(perf['inference_fps'])}；端到端 FPS：{fmt(perf['end_to_end_fps'])}",
             "- 推理耗时包括输入复制、GPU编码器和GPU解码器、输出复制及同步；CUDA事件另外记录GPU计算耗时。",
             "- 端到端 FPS 包括读图、预处理、掩膜恢复/RLE、可选保存和逐帧日志；不含初始化、预热、AP 计算。",
             "- 精度仅使用首轮每张图片一次；重复运行用于持续负载和硬件测量。",
             f"- profiling：{summary['configuration']['profile']}（开启后会影响性能）。",
             "", "| 阶段 ms | 均值 | P50 | P95 | P99 |", "|---|---:|---:|---:|---:|"]
    for stage, values in perf["latency_ms"].items():
        lines.append(f"| {stage} | " + " | ".join(fmt(values[k]) for k in ("mean", "p50", "p95", "p99")) + " |")
    lines += ["", "| 精度（0–1） | AP50 | AP50–95 |", "|---|---:|---:|"]
    for kind in ("bbox", "segm"):
        acc = (summary.get("accuracy") or {}).get(kind, {})
        lines.append(f"| {kind} | {fmt(acc.get('map50'))} | {fmt(acc.get('map50_95'))} |")
    lines += ["", "| 硬件指标 | 均值 | 最大 | 有效样本 |", "|---|---:|---:|---:|"]
    for name, stats in summary["hardware"].get("statistics", {}).items():
        lines.append(f"| {name} | {fmt(stats['mean'])} | {fmt(stats['max'])} | {stats['samples']} |")
    lines += ["", "CPU/GPU/温度/功率轨是整机数据；RSS包括Python和TensorRT，统一内存可能有重叠，不可读项为N/A。",
              "功率按实际传感器供电范围分别记录，不把电压×电流的异步读数或利用率当作整板功率。",
              "能量为有效相邻功率样本的梯形积分，覆盖时长见 summary.json，不外推缺失区间。",
              "没有标注时不计算精度；首轮任一图失败则拒绝部分数据集 AP。",
              "实际TensorRT层信息在engines/精度/*_layers.json；FP16路线包含明示的必要FP32算子。"]
    if summary["configuration"].get("evaluation_box_policy") == "official_coco_raw":
        lines += ["", "预测框评估：有限坐标原值提交官方COCO，与原训练工程一致；不重排端点、不删除反向框。"]
    if summary.get("error"):
        lines += ["", f"错误：{summary['error']}"]
    dump_json(out/"summary.json", summary)
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(args):
    if (args.profile or args.warmup < 0 or args.repeats < 1 or args.cpu_threads < 1 or args.limit < 0 or args.dump_raw < 0
            or args.eval_max_det < 10 or not 0 <= args.conf <= 1 or not 0 <= args.visual_conf <= 1 or not math.isfinite(args.mask_threshold)
            or not math.isfinite(args.sample_interval) or args.sample_interval < 0.02
            or any(not math.isfinite(v) or v < 0 for v in (args.min_seconds, args.baseline_seconds, args.cooldown_seconds))
            or not math.isfinite(args.power_scale) or args.power_scale <= 0
            or getattr(args, "eval_batch_images", 8) < 1):
        raise ValueError("Invalid run parameters: durations>=0, repeats/threads>=1, sample interval>=0.02, conf in [0,1]")
    import cv2
    cv2.setNumThreads(args.cpu_threads)
    manifest = load_manifest(args.manifest)
    names = manifest["classes"]
    image_root = Path(args.images).resolve()
    dataset, mapping = None, manifest["category_ids"]
    if args.coco:
        dataset, images, mapping = load_dataset(image_root, names, coco=args.coco)
        if any(not a.get("segmentation") for a in dataset["annotations"]):
            raise ValueError("COCO instance segmentation evaluation requires segmentation for every annotation")
    else:
        images = collect_images(image_root)
    if args.limit:
        images = images[:args.limit]
        if dataset:
            selected = {p.relative_to(image_root).as_posix() for p in images}
            dataset["images"] = [im for im in dataset["images"] if Path(im["file_name"]).as_posix() in selected]
            ids = {im["id"] for im in dataset["images"]}
            dataset["annotations"] = [a for a in dataset["annotations"] if a["image_id"] in ids]
    output = Path(args.output).resolve() if args.output else ROOT/"runs"/datetime.now().strftime("ecseg_%Y%m%d_%H%M%S_%f")
    if path_within(output, image_root) or path_within(image_root, output):
        raise ValueError("Output and input directories must not contain each other")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty/new")
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "manifest_sha256": sha256(args.manifest), "platform": platform.platform(),
              "python": sys.version, "numpy": np.__version__, "images": str(image_root), "output": str(output),
              "images_selected": len(images), "evaluation_subset": bool(args.limit), "category_mapping": mapping,
              "batch_size": 1, "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    config["evaluation_box_policy"] = "official_coco_raw"
    import pkg_resources
    import tensorrt as trt
    config["packages"] = {name: pkg_resources.get_distribution(name).version for name in ("numpy", "Pillow", "pycocotools")}
    config["packages"]["tensorrt"] = trt.__version__
    config["packages"]["opencv"] = cv2.__version__
    config["source_sha256"] = {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    dump_json(output/"config.json", config)
    dump_json(output/"manifest.json", manifest)
    if args.coco:
        config["annotations_sha256"] = sha256(args.coco)
        dump_json(output/"config.json", config)
        dump_json(output/"ground_truth.coco.json", dataset)
    (output/"images.txt").write_text("\n".join(str(p) for p in images)+"\n", encoding="utf-8")
    if args.save_images:
        (output/"images").mkdir()
    if args.dump_raw:
        (output/"raw").mkdir()
    sampler = Sampler(output, args.sample_interval, args.npu_load_path, args.power_path, args.power_scale)
    sampler.start()
    engine, engine_info = None, {}
    timings = []
    failed, complete_passes, first_pass_failures = 0, 0, 0
    first_pass_complete, fatal, accuracy, accuracy_error = False, None, None, None
    start = end = time.monotonic()
    measurement_started = False
    try:
        board_snapshot(output, "before")
        started_init = time.perf_counter()
        engine = Engine(args, manifest, output)
        engine_info = engine.metadata()
        engine_info["initialization_ms"] = (time.perf_counter()-started_init)*1000
        dump_json(output/"runner_metadata.json", engine_info)
        board_snapshot(output, "initialized")
        sampler.set_phase("baseline")
        time.sleep(args.baseline_seconds)
        sampler.set_phase("warmup")
        for i in range(args.warmup):
            with Image.open(images[i % len(images)]) as image:
                rgb, normalized = preprocess(image, manifest["input_size"])
                outputs, _, _ = engine.infer(rgb, normalized)
                decode(outputs, image.size, names, manifest["num_top_queries"], args.conf, args.mask_threshold)
        print(f"Backend={args.backend}; route={manifest['route']}; classes={names}; images={len(images)}", flush=True)
        sampler.set_phase("measurement")
        start = time.monotonic()
        measurement_started = True
        with (output/"predictions.jsonl").open("w", encoding="utf-8") as predictions, \
             (output/"timings.csv").open("w", encoding="utf-8-sig", newline="") as timing_file:
            writer = None
            pass_index = 0
            while pass_index < args.repeats or time.monotonic()-start < args.min_seconds:
                for index, path in enumerate(images):
                    frame_start = time.perf_counter()
                    row = {"pass": pass_index, "index": index, "image": path.relative_to(image_root).as_posix(),
                           "timestamp_unix_s": time.time(), "timestamp_monotonic_s": time.monotonic(), "status": "ok", "error": ""}
                    result, timing = [], {}
                    try:
                        with Image.open(path) as opened:
                            image = opened.convert("RGB")
                            image.load()
                        read_end = time.perf_counter()
                        rgb, normalized = preprocess(image, manifest["input_size"])
                        prep_end = time.perf_counter()
                        outputs, timing, features = engine.infer(rgb, normalized)
                        post_start = time.perf_counter()
                        result = decode(outputs, image.size, names, manifest["num_top_queries"], args.conf, args.mask_threshold)
                        post_end = time.perf_counter()
                        if pass_index == 0 and index < args.dump_raw:
                            tensors = dict(zip(manifest["feature_names"], features))
                            if not getattr(args, "dump_features_only", False):
                                tensors.update(zip(manifest["output_names"], outputs))
                            np.savez_compressed(output/"raw"/f"{index:06d}.npz", **tensors)
                            dump_json(output/"raw"/f"{index:06d}.json", {"image": row["image"], "image_sha256": sha256(path),
                                      "checkpoint_sha256": manifest["checkpoint_sha256"], "backend": args.backend})
                        if pass_index == 0 and args.save_images:
                            draw(image, [r for r in result if r["score"] > args.visual_conf], output/"images"/f"{index:06d}.png")
                        saved = time.perf_counter()
                        timing.update({"read_ms": (read_end-frame_start)*1000, "preprocess_ms": (prep_end-read_end)*1000,
                                       "postprocess_ms": (post_end-post_start)*1000, "save_ms": (saved-post_end)*1000,
                                       "pipeline_ms": (post_end-read_end)*1000})
                    except Exception as exc:
                        failed += 1
                        first_pass_failures += pass_index == 0
                        row.update(status="failed", error=repr(exc))
                        print(f"FAILED {row['image']}: {exc}", flush=True)
                    if pass_index == 0:
                        predictions.write(json.dumps({**row, "detections": result, "timing": timing}, ensure_ascii=False, allow_nan=False)+"\n")
                        predictions.flush()
                    timing["frame_ms"] = (time.perf_counter()-frame_start)*1000
                    row.update(timing)
                    row["instances"] = len(result)
                    # A fixed union keeps failed frames and optional unsupported driver timings readable.
                    fields = ["pass", "index", "image", "timestamp_unix_s", "timestamp_monotonic_s", "status", "error", "instances",
                              "read_ms", "preprocess_ms", "input_ms", "rknn_run_ms", "output_get_ms", "output_copy_release_ms",
                              "rknn_sync_ms", "rknn_total_ms", "driver_run_ms", "onnx_encoder_ms", "onnx_full_ms", "cpu_decoder_ms", "gpu_encoder_ms", "gpu_decoder_ms", "gpu_compute_ms", "transfer_submit_sync_overhead_ms",
                              "inference_ms", "postprocess_ms", "save_ms", "pipeline_ms", "frame_ms"]
                    if writer is None:
                        writer = csv.DictWriter(timing_file, fieldnames=fields)
                        writer.writeheader()
                    writer.writerow(row)
                    timing_file.flush()
                    timings.append(row)
                pass_index += 1
                complete_passes = pass_index
                first_pass_complete = True
                end = time.monotonic()
                print(f"Pass {pass_index}, frames={len(timings)}, failures={failed}, elapsed={end-start:.2f}s", flush=True)
        end = time.monotonic()
        sampler.set_phase("cooldown")
        if engine.runtime:
            engine_info["memory_after_measurement_bytes"] = engine.runtime.memory_snapshot()
            detail = engine.runtime.perf_detail()
            if detail:
                (output/"rknn_perf_detail.txt").write_text(detail, encoding="utf-8")
            engine_info = {**engine_info, **engine.metadata()}
            dump_json(output/"runner_metadata.json", engine_info)
        board_snapshot(output, "measured")
        time.sleep(args.cooldown_seconds)
    except BaseException as exc:
        fatal = repr(exc)
        end = time.monotonic()
        if not measurement_started:
            start = end
        (output/"fatal_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"ERROR: {exc}", flush=True)
    finally:
        sampler.set_phase("finalization")
        if engine:
            try:
                engine.close()
            except Exception as exc:
                fatal = fatal or f"Finalization failed: {exc!r}"
        sampler.close()
        board_snapshot(output, "after")
    # Persist all measured performance/hardware results before evaluating AP.
    # A native crash or OOM during evaluation must not erase a successful run's report.
    accuracy_ready = dataset is not None and first_pass_complete and not first_pass_failures
    accuracy_status = "not_requested" if dataset is None else "failed"
    if accuracy_ready:
        accuracy_status = "deferred" if getattr(args, "skip_evaluation", False) else "pending"
    good = [row for row in timings if row["status"] == "ok"]
    stages = sorted({key for row in good for key in row if key.endswith("_ms")})
    duration = end-start
    inference_ms = sum(row.get("inference_ms", 0) for row in good)
    performance = {"frames_success": len(good), "frames_failed": failed, "completed_passes": complete_passes,
                   "measurement_duration_s": duration, "inference_fps": len(good)*1000/inference_ms if inference_ms else None,
                   "end_to_end_fps": len(good)/duration if duration > 0 else None,
                   "latency_ms": {stage: percentiles([row[stage] for row in good if isinstance(row.get(stage), (int, float))]) for stage in stages}}
    status = "failed" if fatal or failed or accuracy_status == "failed" else (
        "inference_complete" if accuracy_status == "pending" else "complete")
    summary = {"status": status, "configuration": config, "engine": engine_info, "performance": performance,
               "accuracy_status": accuracy_status, "accuracy": accuracy, "error": fatal or accuracy_error,
               "hardware": sampler.summary(start, end)}
    report(output, summary)
    # Drop references to the last frame and ORT sessions before loading COCO data.
    engine = None
    result = outputs = features = None
    import gc
    gc.collect()
    if accuracy_status == "pending":
        try:
            accuracy = evaluate_prediction_stream(dataset, output/"predictions.jsonl", names, mapping, output,
                        args.eval_max_det, batch_images=getattr(args, "eval_batch_images", 8), box_policy="official_coco_raw")
            accuracy_status = "complete"
        except Exception as exc:
            accuracy_status, accuracy_error = "failed", repr(exc)
            (output/"evaluation_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        summary.update(status="failed" if fatal or failed or accuracy_status == "failed" else "complete",
                       accuracy_status=accuracy_status, accuracy=accuracy, error=fatal or accuracy_error)
        report(output, summary)
    status = summary["status"]
    print(f"status={status}; report={output/'report.md'}", flush=True)
    return 0 if status == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--backend", choices=["tensorrt"], default="tensorrt")
    parser.add_argument("--precision", choices=["fp32","fp16"], required=True)
    parser.add_argument("--engines", required=True)
    parser.add_argument("--model")
    parser.add_argument("--runtime", default="librknnrt.so")
    parser.add_argument("--rknn-input-mode", choices=["uint8", "raw-float32", "native-normalized", "normalized-float32"], default="uint8",
                        help="uint8 baseline; raw-float32 uses 0..255 NHWC; native-normalized bypasses conversion. "
                             "Legacy normalized-float32 is rejected as invalid on this runtime")
    parser.add_argument("--images", required=True)
    parser.add_argument("--coco")
    parser.add_argument("--output")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--visual-conf", type=float, default=0.3)
    parser.add_argument("--mask-threshold", type=float, default=0.0, help="Threshold on resized mask LOGITS")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=0)
    parser.add_argument("--baseline-seconds", type=float, default=2)
    parser.add_argument("--cooldown-seconds", type=float, default=2)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--core-mask", type=int, choices=[0, 1, 2, 3, 4, 7], default=0)
    parser.add_argument("--eval-max-det", type=int, default=100)
    parser.add_argument("--eval-batch-images", type=int, default=8, help="COCO mask-memory batch size; AP is accumulated globally")
    parser.add_argument("--skip-evaluation", action="store_true", help="Save measured results/predictions and defer AP to ecseg_recover_results.py")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dump-raw", type=int, default=0)
    parser.add_argument("--dump-features-only", action="store_true", help="Dump P3/P4/P5 only, reducing diagnostic transfer size")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--npu-load-path")
    parser.add_argument("--power-path")
    parser.add_argument("--power-scale", type=float, default=1)
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
