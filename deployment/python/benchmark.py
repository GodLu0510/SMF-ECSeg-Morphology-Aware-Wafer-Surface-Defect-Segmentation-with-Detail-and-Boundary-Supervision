"""Shared latency statistics; extracted from deployment benchmark helpers."""
import csv
import math
from pathlib import Path


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def percentiles(values):
    if not values:
        return {"mean": None, "min": None, "max": None, "p50": None, "p90": None, "p95": None, "p99": None}
    values = sorted(values)
    def quantile(p):
        idx = (len(values)-1)*p
        lo = int(idx)
        hi = min(lo+1, len(values)-1)
        return values[lo]*(hi-idx)+values[hi]*(idx-lo) if hi != lo else values[lo]
    return {"mean": sum(values)/len(values), "min": values[0], "max": values[-1],
            **{f"p{p}": quantile(p/100) for p in (50, 90, 95, 99)}}
