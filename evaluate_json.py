"""Simple evaluate script for json result file."""
import argparse
import json
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from scripts.utils.logger import get_logger
from scripts.utils.metrics import COCOmAPEvaluator

LOGGER = get_logger(__name__)


def get_args() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--json_path", type=str, help="Prediction result json file for val2017"
    )
    parser.add_argument(
        "--no_coco",
        action="store_true",
        default=False,
        help="Validate with pycocotools.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    gt_path = os.path.join("tests", "res", "instances_val2017.json")

    coco_eval = COCOmAPEvaluator(gt_path)

    json_path = args.json_path
    with open(args.json_path, "r") as f:
        result_json = json.load(f)
    if "framework" in result_json[0].keys():
        result_json = result_json[2:]
        json_path = json_path.rsplit(".", 1)[0] + "_modified.json"
        with open(json_path, "w") as f:
            json.dump(result_json, f)
    result = coco_eval.evaluate(json_path)
    LOGGER.info(f"mAP50: {result['map50']}, mAP50:95: {result['map50_95']}")

    if not args.no_coco:
        anno = COCO(gt_path)
        pred = anno.loadRes(json_path)
        cocotools_eval = COCOeval(anno, pred, "bbox")

        cocotools_eval.evaluate()
        cocotools_eval.accumulate()
        cocotools_eval.summarize()

    os.remove(json_path)
