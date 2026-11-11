"""Bounded-mask-memory evaluation of streamed ECSeg predictions with official COCOeval."""
import contextlib
import copy
import gc
import json
import math
from pathlib import Path

import numpy as np

from ecseg_common import dump_json


def _prediction_images(path, dataset, mapping, kind, box_policy="strict"):
    """Yield one image at a time and refuse duplicate, failed or incomplete first passes."""
    by_name = {Path(image["file_name"]).as_posix(): image for image in dataset["images"]}
    if len(by_name) != len(dataset["images"]):
        raise ValueError("Duplicate ground-truth file names")
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("pass", 0) != 0:
                continue
            name = Path(record["image"]).as_posix()
            if name not in by_name or name in seen or record.get("status") != "ok":
                raise ValueError("Invalid/duplicate/failed prediction image at line %s: %s" % (line_number, name))
            seen.add(name)
            image = by_name[name]
            items = []
            for item in record["detections"]:
                label = item["class_id"]
                if type(label) is not int or not 0 <= label < len(mapping):
                    raise ValueError("Prediction class ID is outside the saved category mapping")
                score = float(item["score"])
                if not math.isfinite(score):
                    raise ValueError("Non-finite prediction confidence")
                result = {"image_id": image["id"], "category_id": mapping[label], "score": score}
                if kind == "bbox":
                    x1, y1, x2, y2 = item["bbox_xyxy"]
                    box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    if not all(math.isfinite(value) for value in box):
                        raise ValueError("Non-finite predicted box at image %s, query %s: %s" %
                                         (name, item.get("query_id"), item["bbox_xyxy"]))
                    if box_policy == "strict" and (box[2] < 0 or box[3] < 0):
                        raise ValueError("Negative predicted box extent at image %s, query %s: %s" %
                                         (name, item.get("query_id"), item["bbox_xyxy"]))
                    result["bbox"] = box
                else:
                    rle = item["segmentation"]
                    if rle["size"] != [image["height"], image["width"]] or not isinstance(rle["counts"], str):
                        raise ValueError("Expected compressed COCO RLE at the original image dimensions")
                    result["segmentation"] = rle
                items.append(result)
            # Release unneeded timing/detection dictionaries before reading another image.
            record = None
            yield image["id"], items
    if seen != set(by_name):
        missing = sorted(set(by_name) - seen)
        raise ValueError("First-pass predictions are incomplete: %s missing images; first=%s" % (len(missing), missing[:3]))


def _batch_result(gt, kind, image_ids, items, params, retained, image_positions, log):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(log):
        if items:
            detected = gt.loadRes(items)
        else:
            detected = COCO()
            detected.dataset = {"images": [gt.imgs[value] for value in image_ids],
                                "categories": gt.dataset["categories"], "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(gt, detected, kind)
        evaluator.params = copy.deepcopy(params)
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
    categories, areas, batch_images = len(params.catIds), len(params.areaRng), len(evaluator.params.imgIds)
    total_images = len(params.imgIds)
    for category in range(categories):
        for area in range(areas):
            for local, image_id in enumerate(evaluator.params.imgIds):
                source = category * areas * batch_images + area * batch_images + local
                target = category * areas * total_images + area * total_images + image_positions[image_id]
                retained[target] = evaluator.evalImgs[source]
    # retained contains only compact matching arrays/scores, never mask RLE strings.


def _metrics(evaluator, dataset, names, mapping, max_dets):
    precision = evaluator.eval["precision"]
    def average(values):
        selected = values[values > -1]
        return float(selected.mean()) if selected.size else None
    per_class = []
    for label, category in enumerate(mapping):
        index = list(evaluator.params.catIds).index(category)
        values = precision[:, :, index, 0, -1]
        per_class.append({"name": names[label], "category_id": category,
                          "map50_95": average(values), "map50": average(values[0]),
                          "ground_truth": sum(item["category_id"] == category and not item.get("iscrowd", 0)
                                              for item in dataset["annotations"])})
    return {"map50_95": average(precision[:, :, :, 0, -1]), "map50": average(precision[0, :, :, 0, -1]),
            "per_class": per_class, "max_dets": max_dets,
            "coco_stats": [float(value) if value >= 0 else None for value in evaluator.stats]}


def evaluate_prediction_stream(dataset, prediction_file, names, mapping, output, max_dets=100,
                               batch_images=8, export_predictions=True, box_policy="strict"):
    """Exact official per-image matching in batches, followed by one global accumulation.

    COCOeval truncates by category/image to maxDets before matching. We apply the
    same stable score ordering before loadRes to avoid retaining irrelevant masks.
    Global AP is accumulated across all images, never averaged across batches.
    """
    # official_coco_raw preserves finite signed widths/heights exactly as the
    # original ECSeg PostProcessor + CocoEvaluator. COCO owns their IoU/area
    # semantics; they are not reordered, clamped, dropped or replaced by masks.
    if box_policy not in ("strict", "official_coco_raw"):
        raise ValueError("Unknown box evaluation policy: " + str(box_policy))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from benchmark import write_csv
    if batch_images < 1 or max_dets < 10:
        raise ValueError("Evaluation batch size must be positive and max_dets at least 10")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    image_ids = sorted(image["id"] for image in dataset["images"])
    if len(image_ids) != len(set(image_ids)) or not image_ids:
        raise ValueError("Empty dataset or duplicate ground-truth image IDs")
    with (output / "coco_evaluation.log").open("w", encoding="utf-8") as log:
        for kind in ("bbox", "segm"):
            gt = COCO()
            gt.dataset = copy.deepcopy(dataset)
            with contextlib.redirect_stdout(log):
                gt.createIndex()
            aggregate = COCOeval(gt, None, kind)
            aggregate.params.imgIds = image_ids
            aggregate.params.catIds = sorted(mapping)
            aggregate.params.maxDets = [1, 10, max_dets]
            params = copy.deepcopy(aggregate.params)
            retained = [None] * (len(params.catIds) * len(params.areaRng) * len(image_ids))
            positions = {image_id: index for index, image_id in enumerate(image_ids)}
            batch_ids, batch_items = [], []
            evaluated, total_predictions, retained_predictions = 0, 0, 0
            destination = output / ("predictions.%s.coco.json" % kind)
            partial = destination.with_suffix(destination.suffix + ".partial")
            writer = partial.open("w", encoding="utf-8") if export_predictions else None
            first_item = True
            complete = False
            try:
                if writer:
                    writer.write("[")
                for image_id, items in _prediction_images(prediction_file, dataset, mapping, kind, box_policy):
                    total_predictions += len(items)
                    if writer:
                        for item in items:
                            if not first_item:
                                writer.write(",\n")
                            json.dump(item, writer, ensure_ascii=False, allow_nan=False)
                            first_item = False
                    counts = {}
                    # Python's stable sort matches COCOeval's stable mergesort for ties.
                    selected = []
                    for item in sorted(items, key=lambda value: -value["score"]):
                        category = item["category_id"]
                        count = counts.get(category, 0)
                        if count < max_dets:
                            selected.append(item)
                            counts[category] = count + 1
                    retained_predictions += len(selected)
                    batch_ids.append(image_id)
                    batch_items.extend(selected)
                    items = selected = None
                    if len(batch_ids) >= batch_images:
                        _batch_result(gt, kind, batch_ids, batch_items, params, retained, positions, log)
                        evaluated += len(batch_ids)
                        batch_ids, batch_items = [], []
                        if writer:
                            writer.flush()
                        print("COCO %s: %s/%s images" % (kind, evaluated, len(image_ids)), flush=True)
                if batch_ids:
                    _batch_result(gt, kind, batch_ids, batch_items, params, retained, positions, log)
                    evaluated += len(batch_ids)
                batch_ids, batch_items = [], []
                if writer:
                    writer.write("]\n")
                    writer.close()
                    writer = None
                    partial.replace(destination)
                complete = True
            finally:
                if writer:
                    writer.close()
                if not complete and partial.exists():
                    partial.unlink()
            aggregate.evalImgs = retained
            aggregate._paramsEval = copy.deepcopy(params)
            with contextlib.redirect_stdout(log):
                aggregate.accumulate()
                aggregate.summarize()
            results[kind] = _metrics(aggregate, dataset, names, mapping, max_dets)
            dump_json(output / "accuracy.partial.json", {"accuracy": results, "completed_tasks": list(results),
                      "evaluation_method": "official COCOeval per-image batches with global accumulation",
                      "last_task_predictions": total_predictions, "last_task_retained_for_matching": retained_predictions})
            write_csv(output / "per_class.csv", [{"task": task, **row} for task, value in results.items() for row in value["per_class"]])
            log.flush()
            del aggregate, retained, gt
            gc.collect()
    dump_json(output / "accuracy.json", results)
    return results
