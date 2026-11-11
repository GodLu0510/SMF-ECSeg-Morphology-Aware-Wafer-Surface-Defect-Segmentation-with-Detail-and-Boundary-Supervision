"""Shared ECSeg contract, image preprocessing and DETR instance decoding."""
import hashlib
import json
from pathlib import Path

SCHEMA = "edgecrafter.ecseg.rk3588.v1"
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]
OUTPUT_NAMES = ["pred_logits", "pred_boxes", "pred_masks"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if data.get("schema") != SCHEMA or data.get("route") not in ("hybrid", "full"):
        raise ValueError("Use the manifest produced by ecseg_export.py")
    if data.get("batch_size") != 1 or not data.get("use_focal_loss"):
        raise ValueError("Only batch=1, focal-loss ECSeg exports are supported")
    return data


def preprocess(image, size):
    """PIL RGB direct resize, same geometry as training/inference; no letterbox."""
    import numpy as np
    from PIL import Image
    height, width = size
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    rgb = np.asarray(image.convert("RGB").resize((width, height), bilinear), dtype=np.uint8).copy()
    normalized = (rgb.astype(np.float32) - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return rgb[None], np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def decode(outputs, original_size, classes, topk=300, confidence=0.001, mask_threshold=0.0):
    """Focal sigmoid + flattened query/class top-k, exactly the ECSeg policy.

    No NMS, no objectness, no YOLO mask prototypes. Bilinear mask LOGITS are
    resized before thresholding. original_size is (width, height).
    """
    import numpy as np
    import cv2
    from pycocotools import mask as mask_util
    logits, boxes, masks = [np.asarray(value, dtype=np.float32) for value in outputs]
    if (logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] != len(classes)
            or boxes.shape != (1, logits.shape[1], 4)
            or masks.ndim != 4 or masks.shape[:2] != logits.shape[:2]):
        raise ValueError(f"Unexpected ECSeg tensors: {[v.shape for v in outputs]}")
    if not all(np.isfinite(value).all() for value in (logits, boxes, masks)):
        raise ValueError("Non-finite model outputs")
    flat = (1.0 / (1.0 + np.exp(-np.clip(logits[0], -80, 80)))).reshape(-1)
    order = np.argsort(-flat, kind="stable")[:min(topk, flat.size)]
    order = order[flat[order] > confidence]
    width, height = original_size
    results, decoded = [], {}
    for index in order:
        query, label = divmod(int(index), len(classes))
        if query not in decoded:
            mask = cv2.resize(masks[0, query], (width, height), interpolation=cv2.INTER_LINEAR) > mask_threshold
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            area = int(mask_util.area(rle))
            rle["counts"] = rle["counts"].decode("ascii")
            decoded[query] = rle, area
        cx, cy, bw, bh = boxes[0, query].astype(float)
        x1, y1, x2, y2 = ((cx-bw/2)*width, (cy-bh/2)*height,
                           (cx+bw/2)*width, (cy+bh/2)*height)
        # Preserve the model's floating boxes for COCO, matching its postprocessor.
        rle, area = decoded[query]
        results.append({"query_id": query, "class_id": label, "class_name": classes[label],
                        "score": float(flat[index]), "bbox_xyxy": [x1, y1, x2, y2],
                        "segmentation": rle, "mask_area_pixels": area})
    return results
