"""Unit test for JSON evaluator.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from scripts.utils.constants import probably_run
from scripts.utils.general import xywh2xyxy
from scripts.utils.metrics import (COCOmAPEvaluator, ap_per_class, box_iou,
                                   check_correct_prediction_by_iou)


@probably_run()
def test_json_evaluator(p=0.5) -> None:
    gt_path = os.path.join("tests", "res", "instances_val2017.json")
    json_path = os.path.join("tests", "res", "answersheet.json")

    coco_eval = COCOmAPEvaluator(gt_path)
    result = coco_eval.evaluate(json_path)

    assert result["map50"] == 0.7526384470506747
    assert result["map50_95"] == 0.5101475752442427


if __name__ == "__main__":
    test_json_evaluator(p=1)
