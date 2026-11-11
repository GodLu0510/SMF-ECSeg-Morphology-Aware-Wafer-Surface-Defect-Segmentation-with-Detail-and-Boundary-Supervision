#!/usr/bin/env python3
"""Compare --dump-raw tensors from identical ONNX and RKNN input images."""
import argparse
import json
from pathlib import Path

import numpy as np

from ecseg_common import dump_json


def compare(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    files = sorted(reference.glob("*.npz"))
    if not files or {p.name for p in files} != {p.name for p in candidate.glob("*.npz")}:
        raise ValueError("Both raw directories must contain the same nonempty set of NPZ files")
    rows = []
    for path in files:
        meta_a = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        meta_b = json.loads((candidate/path.with_suffix(".json").name).read_text(encoding="utf-8"))
        if any(meta_a.get(k) != meta_b.get(k) for k in ("image_sha256", "checkpoint_sha256")):
            raise ValueError(f"Image/checkpoint mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as expected, np.load(candidate/path.name, allow_pickle=False) as actual:
            if set(expected.files) != set(actual.files):
                raise ValueError(f"Tensor names differ: {path.name}")
            for name in expected.files:
                a, b = expected[name].astype(np.float64), actual[name].astype(np.float64)
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f"Invalid shape/numeric output: {path.name}/{name}")
                norm = float(np.linalg.norm(a.ravel())*np.linalg.norm(b.ravel()))
                rows.append({"image": meta_a["image"], "tensor": name, "shape": list(a.shape),
                             "mean_abs_error": float(np.mean(np.abs(a-b))),
                             "max_abs_error": float(np.max(np.abs(a-b))),
                             "rmse": float(np.sqrt(np.mean((a-b)**2))),
                             "cosine_similarity": float(np.dot(a.ravel(), b.ravel())/norm) if norm else None})
    return {"comparisons": rows, "scope": "same-index tensors; query rankings may change after quantization",
            "note": "Numerical diagnostics do not replace bbox+segm AP evaluation on the full validation set"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Choose a new output JSON")
    dump_json(output, compare(args.reference, args.candidate))


if __name__ == "__main__":
    main()
