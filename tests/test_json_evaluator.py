"""Unit test for JSON evaluator.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import json
import os
import random

import numpy as np

from scripts.utils.metrics import bbox_iou


def test_json_evaluator(p: float = 0.5) -> None:
    if random.random() > p:
        return

    label_root = os.path.join("tests", "res", "datasets", "coco", "labels", "val2017")
    json_path = os.path.join("tests", "res", "answersheet.json")
    with open(json_path, "r") as f:
        preds = json.load(f)

    unique_id = set([pred["image_id"] for pred in preds])
    for img_id in unique_id:
        label_name = f"{img_id:012d}.txt"
        label_path = os.path.join(label_root, label_name)

        labels = [pred for pred in preds if pred["image_id"] == img_id]

        if not os.path.isfile(label_path):
            continue

        label_pred = np.array(
            [[label["category_id"], *label["bbox"], label["score"]] for label in labels]
        )
        label_gt = np.loadtxt(label_path)
        import pdb

        pdb.set_trace()


if __name__ == "__main__":
    test_json_evaluator(p=1.0)
