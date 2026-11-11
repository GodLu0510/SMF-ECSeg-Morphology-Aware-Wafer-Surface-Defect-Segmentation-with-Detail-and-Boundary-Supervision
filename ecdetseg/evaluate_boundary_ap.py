
"""Evaluate saved COCO mask predictions using mask AP and Boundary AP."""
import argparse
import json
from pathlib import Path
from boundary_ap import load_ground_truth, evaluate_predictions

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', required=True)
    p.add_argument('--predictions', required=True, help='COCO segmentation prediction JSON')
    p.add_argument('--output', required=True)
    p.add_argument('--dilation-ratio', type=float, default=0.02)
    a = p.parse_args()
    truth = load_ground_truth(a.annotations)
    prediction = json.loads(Path(a.predictions).read_text(encoding='utf-8-sig'))
    if not isinstance(prediction, list):
        p.error('Predictions must be a COCO result list.')
    metrics = evaluate_predictions(truth, prediction, dilation_ratio=a.dilation_ratio)
    destination = Path(a.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
