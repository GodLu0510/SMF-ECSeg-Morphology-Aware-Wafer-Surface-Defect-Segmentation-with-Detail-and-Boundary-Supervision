#!/usr/bin/env python3
"""Recover saved ECSeg metrics/reports without running either ONNX or RKNN inference."""
import argparse
from collections import Counter
import csv
from datetime import datetime
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys

from benchmark import percentiles
from ecseg_benchmark import report
from ecseg_common import dump_json, sha256
from ecseg_evaluation import evaluate_prediction_stream
from telemetry import summarize_samples


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def summarize_saved_run(run, accuracy=None):
    """Reconstruct timing/hardware summaries using captured board data, never PC timing."""
    run = Path(run).resolve()
    config, manifest = read_json(run / "config.json"), read_json(run / "manifest.json")
    engine = read_json(run / "runner_metadata.json")
    dataset = read_json(run / "ground_truth.coco.json")
    with (run / "timings.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Saved timings are empty")
    expected = {Path(image["file_name"]).as_posix() for image in dataset["images"]}
    first = [row for row in rows if int(row["pass"]) == 0]
    if (len(first) != len(expected) or {row["image"] for row in first} != expected
            or any(row["status"] != "ok" for row in first)):
        raise ValueError("The first inference pass was incomplete/failed; refusing partial-dataset AP")
    good = [row for row in rows if row["status"] == "ok"]
    stages = sorted(key for key in rows[0] if key.endswith("_ms") and any(row.get(key) for row in good))
    def numeric(row, key):
        try:
            value = float(row[key])
            return value if math.isfinite(value) else None
        except (ValueError, TypeError, KeyError):
            return None
    start = min(float(row["timestamp_monotonic_s"]) for row in rows)
    end = max(float(row["timestamp_monotonic_s"]) + float(row.get("frame_ms") or 0) / 1000 for row in rows)
    duration_source = "reconstructed from first/last per-frame monotonic times; tiny start/end bookkeeping excluded"
    original_summary = run / "summary.json"
    if original_summary.is_file():
        prior = read_json(original_summary)
        duration = prior["performance"]["measurement_duration_s"]
        duration_source = "original completed summary; hardware interval starts at first saved frame"
        end = start + duration
    else:
        duration = end - start
    total_inference = sum(float(row.get("inference_ms") or 0) for row in good)
    passes = Counter(int(row["pass"]) for row in rows)
    performance = {"frames_success": len(good), "frames_failed": len(rows) - len(good),
                   "completed_passes": sum(count == len(expected) for count in passes.values()),
                   "measurement_duration_s": duration,
                   "inference_fps": len(good) * 1000 / total_inference if total_inference else None,
                   "end_to_end_fps": len(good) / duration if duration > 0 else None,
                   "latency_ms": {key: percentiles([value for row in good if (value := numeric(row, key)) is not None])
                                  for key in stages}}
    samples = []
    with (run / "hardware_samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row["timestamp_unix_s"] = row.pop("timestamp_monotonic_s")
            row.pop("phase", None)
            for key in list(row):
                if key.endswith("_raw") and key != "npu_load_raw":
                    row.pop(key)
            samples.append(row)
    hardware = summarize_samples(samples, start, end)
    hardware.update(scope="captured RK3588 device/process samples during inference; excludes the later AP/OOM stage",
                    time_basis="monotonic", sampler_errors=len(read_json(run / "telemetry_errors.json")))
    failed = performance["frames_failed"] > 0
    summary = {"status": "failed" if failed else ("complete" if accuracy else "inference_complete"),
               "configuration": config, "engine": engine, "performance": performance,
               "accuracy_status": "complete" if accuracy else "pending", "accuracy": accuracy,
               "error": None, "hardware": hardware,
               "recovery": {"source_run": str(run), "model_inference_repeated": False,
                            "duration_source": duration_source, "accuracy_evaluation_host": platform.platform(),
                            "accuracy_pycocotools": metadata.version("pycocotools"),
                            "source_sha256": {name: sha256(run / name) for name in
                                ("config.json", "manifest.json", "ground_truth.coco.json", "timings.csv", "predictions.jsonl")}}}
    return summary, dataset, manifest


def write_recovered_report(output, summary):
    report(output, summary)
    with (Path(output) / "report.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## 从已保存数据恢复\n\n本报告没有重跑模型。性能和硬件数据来自原开发板运行；"
                     "AP从完整首轮预测恢复。\n\n耗时区间来源：" + summary["recovery"]["duration_source"]
                     + "\n\n硬件采样已在AP计算前结束，因此不包含随后评估阶段的内存峰值。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing original run folder")
    parser.add_argument("--output", help="New folder; original files are never overwritten")
    parser.add_argument("--eval-batch-images", type=int, default=8)
    parser.add_argument("--performance-only", action="store_true")
    parser.add_argument("--export-coco", action="store_true", help="Also export potentially large COCO result JSON files")
    args = parser.parse_args()
    run = Path(args.run).resolve()
    output = Path(args.output).resolve() if args.output else run.with_name(run.name + datetime.now().strftime("_recovered_%Y%m%d_%H%M%S"))
    if output == run or output.exists():
        parser.error("Choose a new output folder; never overwrite original results")
    output.mkdir(parents=True)
    summary, dataset, manifest = summarize_saved_run(run)
    write_recovered_report(output, summary)
    if not args.performance_only:
        accuracy = evaluate_prediction_stream(dataset, run / "predictions.jsonl", manifest["classes"],
                    summary["configuration"]["category_mapping"], output / "evaluation",
                    summary["configuration"].get("eval_max_det", 100), args.eval_batch_images,
                    export_predictions=args.export_coco)
        summary.update(status="failed" if summary["performance"]["frames_failed"] else "complete",
                       accuracy_status="complete", accuracy=accuracy)
        write_recovered_report(output, summary)
        shutil.copy2(output / "evaluation/per_class.csv", output / "per_class.csv")
    print("Recovered report:", output / "report.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
