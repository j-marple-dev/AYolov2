"""Ensemble prediction results.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""
import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import tqdm
import yaml
from lib.nms_utils import (nms, non_maximum_weighted, soft_nms,
                           weighted_boxes_fusion)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ensemble-cfg",
        type=str,
        default="configs/ensemble_config.yaml",
        help="config file for ensemble",
    )
    return parser.parse_args()


def load_preds(preds_paths: List, img_shapes: Dict) -> Tuple[str, int, Dict]:
    """Load prediction results.

    Args:
        preds_paths: json paths of prediction results
        img_shapes: original image shapes

    Returns:
        framework: framework of models
        n_params: the number of parameters of all models that wants to ensemble
        preds_dict: prediction results of all models that wants to ensemble
    """
    framework, n_params, preds_dict = "", 0, {}
    for img_id, img_shape in img_shapes.items():
        preds_dict[img_id] = {
            "bboxes_list": [[np.zeros((0, 4))] for _ in range(len(preds_paths))],
            "scores_list": [[] for _ in range(len(preds_paths))],
            "labels_list": [[] for _ in range(len(preds_paths))],
            "image_shape": img_shape,
        }

    print("Load predictions:")
    for model_id, pred_path in enumerate(preds_paths):
        with open(pred_path, "r") as f:
            pred = json.load(f)

        if not framework:
            framework = pred[0]["framework"]
        n_params += pred[1]["parameters"]

        for bbox_info in tqdm.tqdm(pred[2:], desc=f"  Model {model_id}"):
            img_id = str(bbox_info["image_id"])
            img_h, img_w = img_shapes[img_id]
            bbox = np.array(bbox_info["bbox"])
            bbox[2:4] += bbox[0:2]
            bbox = bbox / np.array([img_w, img_h] * 2)
            bbox = np.clip(bbox, 0, 1)[np.newaxis, :]
            preds_dict[img_id]["bboxes_list"][model_id].append(bbox)
            preds_dict[img_id]["scores_list"][model_id].append(bbox_info["score"])
            preds_dict[img_id]["labels_list"][model_id].append(bbox_info["category_id"])

    return framework, n_params, preds_dict


def apply_ensemble(
    framework: str,
    n_params: int,
    preds_dict: Dict,
    img_shapes: Dict,
    nms_type: str,
    weights: np.ndarray,
    iou_thr: float,
    skip_box_thr: float,
    sigma: float,
) -> List:
    """Apply ensemble with prediction results.

    Args:
        framework: framework of models
        n_params: the number of parameters of all models that wants to ensemble
        preds_dict: prediction results of all models that wants to ensemble
        img_shapes: original image shapes
        nms_type: NMS types (e.g. nms, soft_nms, nmw, wbf)
        weights: list of weights for each model. Default: None, which means weight == 1 for each model
        iou_thr: IoU value for boxes to be a match
        skip_box_thr: threshold for boxes to keep (important for SoftNMS)
        sigma: Sigma value for SoftNMS

    Returns:
        preds_list: ensembled prediction results
    """
    preds_list = [
        {"framework": framework},
        {"parameters": n_params},
    ]
    print(f"# of parameters: {n_params:,d}")
    for img_id, preds in tqdm.tqdm(preds_dict.items(), desc="Apply ensemble"):
        preds["bboxes_list"] = [np.concatenate(b) for b in preds["bboxes_list"]]
        if nms_type == "nms":
            boxes, scores, labels = nms(
                preds["bboxes_list"],
                preds["scores_list"],
                preds["labels_list"],
                weights=weights,
                iou_thr=iou_thr,
            )
        elif nms_type == "soft_nms":
            boxes, scores, labels = soft_nms(
                preds["bboxes_list"],
                preds["scores_list"],
                preds["labels_list"],
                weights=weights,
                iou_thr=iou_thr,
                sigma=sigma,
                thresh=skip_box_thr,
            )
        elif nms_type == "nmw":
            boxes, scores, labels = non_maximum_weighted(
                preds["bboxes_list"],
                preds["scores_list"],
                preds["labels_list"],
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        elif nms_type == "wbf":
            boxes, scores, labels = weighted_boxes_fusion(
                preds["bboxes_list"],
                preds["scores_list"],
                preds["labels_list"],
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        else:
            assert "Wrong NMS type!!"

        for box, score, label in zip(boxes, scores, labels):
            img_h, img_w = img_shapes[img_id]
            box[2:4] -= box[0:2]
            box *= np.array([img_w, img_h] * 2)
            preds_list.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": box.tolist(),
                    "score": float(score),
                }
            )
    return preds_list


if __name__ == "__main__":
    ANSWER_PATH = "answersheet_4_04_jmarple.json"
    args = get_parser()
    with open(args.ensemble_cfg, "r") as f:
        ensemble_cfg = yaml.safe_load(f)
    with open(ensemble_cfg["preds"]["img_shape_json"], "r") as f:
        img_shapes = json.load(f)

    framework, n_params, preds_dict = load_preds(
        ensemble_cfg["preds"]["preds_paths"], img_shapes
    )
    preds = apply_ensemble(
        framework,
        n_params,
        preds_dict,
        img_shapes,
        ensemble_cfg["ensemble"]["nms_type"],
        np.array(ensemble_cfg["ensemble"]["weights"]),
        ensemble_cfg["ensemble"]["iou_thr"],
        ensemble_cfg["ensemble"]["skip_box_thr"],
        ensemble_cfg["ensemble"]["sigma"],
    )
    with open(ANSWER_PATH, "w") as f:
        json.dump(preds, f)
