"""Strict dataset loading and official COCO bbox AP evaluation."""
import contextlib
import io
import json
import math
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

def path_within(path, root):
    """Path.is_relative_to equivalent compatible with Python 3.8."""
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def class_names(path):
    names = Path(path).read_text(encoding="utf-8-sig").splitlines()
    names = [name.strip() for name in names]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Class names must be nonempty and unique, one name per line")
    return names


def collect_images(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    images = sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No JPG/JPEG/PNG images in {root}")
    if len(set(images)) != len(images):
        raise ValueError("Duplicate images via symbolic links")
    return images


def image_size(path):
    from PIL import Image
    with Image.open(path) as image:
        # Native decoding does not apply EXIF rotation; annotations use stored pixels.
        return image.size


def load_dataset(images_root, names, labels=None, coco=None, category_ids=None):
    """Returns (COCO dataset, exact selected image paths, model-index to category-id)."""
    root = Path(images_root).resolve()
    if (labels is None) == (coco is None):
        raise ValueError("Supply exactly one of --labels or --coco")
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    if coco:
        data = json.loads(Path(coco).read_text(encoding="utf-8-sig"))
        categories = data["categories"]
        if len({c["id"] for c in categories}) != len(categories):
            raise ValueError("Duplicate COCO category IDs")
        if category_ids:
            mapping = [int(value) for value in category_ids.split(",")]
        else:
            by_name = {c["name"]: c["id"] for c in categories}
            if len(by_name) != len(categories):
                raise ValueError("Duplicate category names; supply --category-ids")
            if any(name not in by_name for name in names):
                raise ValueError("COCO/model class names differ; supply --category-ids in model class order")
            mapping = [by_name[name] for name in names]
        if len(mapping) != len(names) or len(set(mapping)) != len(mapping) or set(mapping) != {c["id"] for c in categories}:
            raise ValueError("Category mapping must cover every COCO category exactly once")
        paths, ids = [], set()
        for image in data["images"]:
            if image["id"] in ids:
                raise ValueError("Duplicate COCO image IDs")
            ids.add(image["id"])
            path = (root / image["file_name"]).resolve()
            if not path_within(path, root) or not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Invalid/missing COCO image: {image['file_name']}")
            width, height = image_size(path)
            if (width, height) != (image["width"], image["height"]):
                raise ValueError(f"COCO image dimensions differ from stored pixels: {path}")
            paths.append(path)
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Empty COCO image set or duplicate file paths")
        ann_ids = set()
        for ann in data["annotations"]:
            if ann["id"] in ann_ids or ann["image_id"] not in ids or ann["category_id"] not in mapping:
                raise ValueError("Invalid/duplicate annotation IDs or unknown image/category")
            ann_ids.add(ann["id"])
            box = ann["bbox"]
            if len(box) != 4 or not all(math.isfinite(x) for x in box) or box[2] <= 0 or box[3] <= 0:
                raise ValueError("Invalid COCO bbox (expected finite x,y,w,h with positive w/h)")
            if ann.get("iscrowd", 0) not in (0, 1):
                raise ValueError("iscrowd must be 0 or 1")
            ann.setdefault("iscrowd", 0)
            ann.setdefault("area", box[2] * box[3])
        if not any(not a["iscrowd"] for a in data["annotations"]):
            raise ValueError("No non-crowd ground-truth objects; mAP is undefined")
        data.setdefault("info", {})
        return data, paths, mapping

    paths = collect_images(root)
    label_root = Path(labels).resolve()
    if not label_root.is_dir():
        raise ValueError(f"Label directory does not exist: {label_root}")
    data = {"info": {}, "images": [], "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(names)]}
    seen_labels = set()
    for image_id, path in enumerate(paths, 1):
        relative = path.relative_to(root)
        label_path = label_root / relative.with_suffix(".txt")
        if label_path in seen_labels:
            raise ValueError(f"Images with the same stem share a label file: {label_path}")
        seen_labels.add(label_path)
        if not label_path.is_file():
            raise ValueError(f"Missing label: {label_path}. Negative images require an empty TXT file.")
        width, height = image_size(path)
        data["images"].append({"id": image_id, "file_name": relative.as_posix(), "width": width, "height": height})
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_no}: expected class xc yc w h (detection labels only)")
            cls, xc, yc, w, h = map(float, fields)
            if not all(math.isfinite(x) for x in (cls, xc, yc, w, h)):
                raise ValueError(f"{label_path}:{line_no}: non-finite label")
            if cls != int(cls) or not 0 <= cls < len(names):
                raise ValueError(f"{label_path}:{line_no}: invalid class ID")
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f"{label_path}:{line_no}: YOLO coordinates must be normalized to [0,1]")
            if xc-w/2 < -1e-5 or yc-h/2 < -1e-5 or xc+w/2 > 1+1e-5 or yc+h/2 > 1+1e-5:
                raise ValueError(f"{label_path}:{line_no}: bbox extends outside image")
            bbox = [(xc-w/2)*width, (yc-h/2)*height, w*width, h*height]
            data["annotations"].append({"id": len(data["annotations"])+1, "image_id": image_id,
                "category_id": int(cls), "bbox": bbox, "area": bbox[2]*bbox[3], "iscrowd": 0})
    if not data["annotations"]:
        raise ValueError("No ground-truth objects; mAP is undefined")
    return data, paths, list(range(len(names)))


def read_predictions(path):
    records = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prediction JSONL at line {line_no}: {exc}") from exc
    if not records:
        raise ValueError("Empty prediction file")
    return records


def evaluate(records, data, paths, mapping, max_det=100):
    """COCO's 101 recall points, IoUs .50:.05:.95, class-aware one-to-one matching."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    if max_det not in (100, 300):
        raise ValueError("Evaluation max_det must be 100 or 300")
    expected = {str(path.resolve()): image for path, image in zip(paths, data["images"])}
    relative_paths = {image["file_name"]: str(path.resolve()) for path, image in zip(paths, data["images"])}
    seen, results = set(), []
    for record in records:
        if "image_relative" in record:
            relative = record["image_relative"]
            if relative not in relative_paths:
                raise ValueError(f"Unexpected prediction relative image: {relative}")
            key = relative_paths[relative]
        else:
            key = str(Path(record["image"]).resolve())
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected/duplicate prediction image: {key}")
        seen.add(key)
        if record["status"] != "ok":
            raise ValueError(f"Image inference failed; refusing partial mAP: {key}")
        image = expected[key]
        if (record["width"], record["height"]) != (image["width"], image["height"]):
            raise ValueError(f"Prediction/image dimensions differ: {key}")
        for det in record["detections"]:
            cls = det["class_id"]
            score = det["score"]
            box = det["bbox"]
            if not isinstance(cls, int) or not 0 <= cls < len(mapping):
                raise ValueError(f"Invalid prediction class ID: {cls}")
            if len(box) != 4 or not all(math.isfinite(x) for x in box) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Non-finite/invalid prediction bbox or score")
            x1, y1, x2, y2 = box
            if not (0 <= x1 <= x2 <= image["width"] and 0 <= y1 <= y2 <= image["height"]):
                raise ValueError("Prediction bbox is outside the image")
            if x2 == x1 or y2 == y1:
                continue
            results.append({"image_id": image["id"], "category_id": mapping[cls],
                            "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
    if seen != set(expected):
        raise ValueError(f"Missing predictions for {len(set(expected)-seen)} images; refusing partial mAP")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = data
        gt.createIndex()
        if results:
            dt = gt.loadRes(results)
        else:
            # COCO.loadRes([]) indexes the first result and fails.
            dt = COCO()
            dt.dataset = {"images": data["images"], "categories": data["categories"], "annotations": []}
            dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = [image["id"] for image in data["images"]]
        ev.params.catIds = list(mapping)
        ev.params.maxDets = [1, 10, max_det]
        ev.evaluate()
        ev.accumulate()
    precision = ev.eval["precision"][:, :, :, 0, -1]
    recall = ev.eval["recall"][:, :, 0, -1]

    def valid_mean(values):
        values = values[values >= 0]
        return float(np.mean(values)) if values.size else None

    per_class = []
    names_by_id = {c["id"]: c["name"] for c in data["categories"]}
    # COCOeval sorts category IDs internally; model order can differ.
    for cls, cat in enumerate(mapping):
        k = list(ev.params.catIds).index(cat)
        count = sum(a["category_id"] == cat and not a.get("iscrowd", 0) for a in data["annotations"])
        per_class.append({"class_id": cls, "category_id": cat, "name": names_by_id[cat], "ground_truth": count,
            "map50": valid_mean(precision[0, :, k]), "map50_95": valid_mean(precision[:, :, k])})
    metrics = {"map50": valid_mean(precision[0]), "map50_95": valid_mean(precision),
               "recall50": valid_mean(recall[0]), "recall50_95": valid_mean(recall),
               "evaluator": "pycocotools.COCOeval bbox", "iou_thresholds": "0.50:0.05:0.95",
               "max_detections_per_image": max_det, "images": len(paths),
               "ground_truth": len(data["annotations"]), "per_class": per_class}
    return metrics, results
