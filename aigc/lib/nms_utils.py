"""Non Maximum Supression module.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision
from numba import jit


@torch.no_grad()
def batched_nms(
    prediction: torch.Tensor,
    conf_thres: float = 0.001,
    iou_thres: float = 0.65,
    nms_box: int = 500,
    agnostic: bool = False,
) -> List[torch.Tensor]:
    """Batched non max supression for faster run.

    Args:
        prediction: batch out (batch, n_obj, 4+1+n_class)
        conf_thres: confidence threshold.
        iou_thres: IoU threshold.
        nms_box: Number of boxes to use before check confidecne threshold.
        agnostic: Separate bboxes by classes for NMS with class separation.

    Return:
        List of NMS filtered result
         - Length of list: batch
         - Elements: (n_obj, 6) (x1, y1, x2, y2, confidence, class_id)
    """
    batch_size = prediction.shape[0]

    # Filter by object score
    out_idx = prediction[:, :, 4].argsort(descending=True)[:, :nms_box]
    out = torch.stack([prediction[i, out_idx[i]] for i in range(batch_size)])

    # Filter by confidecne
    confs = out[:, :, 5:] * out[:, :, 4:5]
    batch_idx, j, k = (confs > conf_thres).nonzero(as_tuple=False).T
    x = torch.cat((out[batch_idx, j, :4], confs[batch_idx, j, k, None], k[:, None]), 1)

    # Batch xywh to xyxy
    xywh = x[:, :4].clone()
    x[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    x[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    x[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    x[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0

    outputs = [x[batch_idx == i] for i in range(batch_size)]
    for i in range(batch_size):
        if agnostic:
            box_offset = outputs[i][:, 5:6] * 1280  # Separate different classes.
            bboxes = outputs[i][:, :4] + box_offset
        else:
            bboxes = outputs[i][:, :4]

        nms_idx = torchvision.ops.nms(bboxes, outputs[i][:, 4], iou_thres)
        outputs[i] = outputs[i][nms_idx]

    return outputs


def prepare_boxes(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare bounding boxes, scores, and labels.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_nms.py

    Args:
        boxes: list of boxes predictions from each model, each box is 4 numbers.
               It has 3 dimensions (models_number, model_preds, 4)
               Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
        scores: list of scores for each model
        labels: list of labels for each model

    Returns:
        result_boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
        scores: confidence scores
        labels: boxes labels
    """
    result_boxes = boxes.copy()

    cond = result_boxes < 0
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print("Warning. Fixed {} boxes coordinates < 0".format(cond_sum))
        result_boxes[cond] = 0

    cond = result_boxes > 1
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print(
            "Warning. Fixed {} boxes coordinates > 1. Check that your boxes was normalized at [0, 1]".format(
                cond_sum
            )
        )
        result_boxes[cond] = 1

    boxes1 = result_boxes.copy()
    result_boxes[:, 0] = np.min(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 2] = np.max(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 1] = np.min(boxes1[:, [1, 3]], axis=1)
    result_boxes[:, 3] = np.max(boxes1[:, [1, 3]], axis=1)

    area = (result_boxes[:, 2] - result_boxes[:, 0]) * (
        result_boxes[:, 3] - result_boxes[:, 1]
    )
    cond = area == 0
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print("Warning. Removed {} boxes with zero area!".format(cond_sum))
        result_boxes = result_boxes[area > 0]
        scores = scores[area > 0]
        labels = labels[area > 0]

    return result_boxes, scores, labels


def cpu_soft_nms_float(
    dets: np.ndarray,
    sc: np.ndarray,
    Nt: float,
    sigma: float,
    thresh: float,
    method: int,
) -> np.ndarray:
    """Soft NMS with cpu.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_nms.py

    Based on: https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py.
    It's different from original soft-NMS because we have float coordinates on range [0; 1].

    Args:
        dets:   boxes format [x1, y1, x2, y2]
        sc:     scores for boxes
        Nt:     required iou
        sigma: Sigma value for SoftNMS
        thresh: threshold for boxes to keep (important for SoftNMS)
        method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS

    Returns:
        keep: index of boxes to keep
    """
    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1, x1, y2, x2]
    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = dets[:, 3]
    x2 = dets[:, 2]
    scores = sc
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)
    return keep


@jit(nopython=True)
def nms_float_fast(dets: np.ndarray, scores: np.ndarray, thresh: float) -> List:
    """Fast NMS function.

       It's different from original nms because we have float coordinates on range [0; 1].
       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_nms.py

    Args:
        dets: numpy array of boxes with shape: (N, 5). Order: x1, y1, x2, y2, score. All variables in range [0; 1]
        thresh: IoU value for boxes

    Returns:
        keep: index of boxes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_method(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    method: int = 3,
    iou_thr: float = 0.5,
    sigma: float = 0.5,
    thresh: float = 0.001,
    weights: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NMS method for hard NMS and soft NMS.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_nms.py

    Args:
        boxes: list of boxes predictions from each model, each box is 4 numbers.
               It has 3 dimensions (models_number, model_preds, 4)
               Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
        scores: list of scores for each model
        labels: list of labels for each model
        method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
        iou_thr: IoU value for boxes to be a match
        sigma: Sigma value for SoftNMS
        thresh: threshold for boxes to keep (important for SoftNMS)
        weights: list of weights for each model. Default: None, which means weight == 1 for each model

    Returns:
        final_boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
        final_scores: confidence scores
        final_labels: boxes labels
    """
    # If weights are specified
    if weights is not None:
        if len(boxes) != len(weights):
            print(
                "Incorrect number of weights: {}. Must be: {}. Skip it".format(
                    len(weights), len(boxes)
                )
            )
        else:
            weights = np.array(weights)
            for i in range(len(weights)):
                scores[i] = (np.array(scores[i]) * weights[i]) / weights.sum()

    # We concatenate everything
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    # Fix coordinates and removed zero area boxes
    boxes, scores, labels = prepare_boxes(boxes, scores, labels)

    # Run NMS independently for each label
    unique_labels = np.unique(labels)
    final_boxes_list = []
    final_scores_list = []
    final_labels_list = []
    for unique_label in unique_labels:
        condition = labels == unique_label
        boxes_by_label = boxes[condition]
        scores_by_label = scores[condition]
        labels_by_label = np.array([unique_label] * len(boxes_by_label))

        if method != 3:
            keep = cpu_soft_nms_float(
                boxes_by_label.copy(),
                scores_by_label.copy(),
                Nt=iou_thr,
                sigma=sigma,
                thresh=thresh,
                method=method,
            )
        else:
            # Use faster function
            keep = nms_float_fast(boxes_by_label, scores_by_label, thresh=iou_thr)

        final_boxes_list.append(boxes_by_label[keep])
        final_scores_list.append(scores_by_label[keep])
        final_labels_list.append(labels_by_label[keep])

    final_boxes = np.concatenate(final_boxes_list)
    final_scores = np.concatenate(final_scores_list)
    final_labels = np.concatenate(final_labels_list)

    return final_boxes, final_scores, final_labels


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_thr: float = 0.5,
    weights: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short call for standard NMS.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_nms.py

    Args:
        boxes: list of boxes predictions from each model, each box is 4 numbers.
               It has 3 dimensions (models_number, model_preds, 4)
               Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
        scores: list of scores for each model
        labels: list of labels for each model
        iou_thr: IoU value for boxes to be a match
        weights: list of weights for each model. Default: None, which means weight == 1 for each model

    Returns:
        boxes coordinates (Order of boxes: x1, y1, x2, y2).
        confidence scores
        boxes labels
    """
    return nms_method(boxes, scores, labels, method=3, iou_thr=iou_thr, weights=weights)


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    method: int = 2,
    iou_thr: float = 0.5,
    sigma: float = 0.5,
    thresh: float = 0.001,
    weights: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short call for Soft-NMS.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_nms.py

    Args:
        boxes: list of boxes predictions from each model, each box is 4 numbers.
               It has 3 dimensions (models_number, model_preds, 4)
               Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
        scores: list of scores for each model
        labels: list of labels for each model
        method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
        iou_thr: IoU value for boxes to be a match
        sigma: Sigma value for SoftNMS
        thresh: threshold for boxes to keep (important for SoftNMS)
        weights: list of weights for each model. Default: None, which means weight == 1 for each model

    Returns:
        boxes coordinates (Order of boxes: x1, y1, x2, y2).
        confidence scores
        boxes labels
    """
    return nms_method(
        boxes,
        scores,
        labels,
        method=method,
        iou_thr=iou_thr,
        sigma=sigma,
        thresh=thresh,
        weights=weights,
    )


@jit(nopython=True)
def bb_intersection_over_union(A: List, B: List) -> float:
    """Calculate IoU.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py

    Args:
        A: list of bounding boxes
        B: list of bounding boxes

    Returns:
        iou: IoU between tow bounding boxes
    """
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    thr: float,
) -> Dict:
    """Filter bounding boxes, scores, and labels.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py

    Args:
        boxes: list of boxes predictions from each model, each box is 4 numbers.
               It has 3 dimensions (models_number, model_preds, 4)
               Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
        scores: list of scores for each model
        labels: list of labels for each model
        weights: list of weights for each model. Default: None, which means weight == 1 for each model
        thr: threshold for boxes to keep (important for SoftNMS)

    Returns:
        new_boxes: filtered bounding boxes
    """
    # Create dict with boxes stored by its label
    new_boxes: Dict[int, List] = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print(
                "Error. Length of boxes arrays not equal to length of scores array: {} != {}".format(
                    len(boxes[t]), len(scores[t])
                )
            )
            exit()

        if len(boxes[t]) != len(labels[t]):
            print(
                "Error. Length of boxes arrays not equal to length of labels array: {} != {}".format(
                    len(boxes[t]), len(labels[t])
                )
            )
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn("X2 < X1 value in box. Swap them.")
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn("Y2 < Y1 value in box. Swap them.")
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn("X1 < 0 in box. Set it to 0.")
                x1 = 0
            if x1 > 1:
                warnings.warn(
                    "X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range."
                )
                x1 = 1
            if x2 < 0:
                warnings.warn("X2 < 0 in box. Set it to 0.")
                x2 = 0
            if x2 > 1:
                warnings.warn(
                    "X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range."
                )
                x2 = 1
            if y1 < 0:
                warnings.warn("Y1 < 0 in box. Set it to 0.")
                y1 = 0
            if y1 > 1:
                warnings.warn(
                    "Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range."
                )
                y1 = 1
            if y2 < 0:
                warnings.warn("Y2 < 0 in box. Set it to 0.")
                y2 = 0
            if y2 > 1:
                warnings.warn(
                    "Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range."
                )
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes: np.ndarray, conf_type: str = "avg") -> np.ndarray:
    """Create weighted box for set of boxes.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py

    Args:
        boxes: set of boxes to fuse
        conf_type: type of confidence one of 'avg' or 'max'

    Returns:
        weighted box (label, score, weight, x1, y1, x2, y2)
    """
    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += b[1] * b[4:]
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type == "avg":
        box[1] = conf / len(boxes)
    elif conf_type == "max":
        box[1] = np.array(conf_list).max()
    elif conf_type in ["box_and_model_avg", "absent_model_aware_avg"]:
        box[1] = conf / len(boxes)
    box[2] = w
    box[3] = -1  # model index field is retained for consistensy but is not used.
    box[4:] /= conf
    return box


def find_matching_box(
    boxes_list: List, new_box: List, match_iou: float
) -> Tuple[int, float]:
    """Find matching bounding boxes.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py

    Args:
        boxes_list: list of bounding boxes
        new_box: new bounding box
        match_iou: IoU threshold

    Returns:
        best_index: index of best matched bounding box
        best_iou: best IoU of best matched bounding box
    """
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[4:], new_box[4:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(
    boxes_list: np.ndarray,
    scores_list: np.ndarray,
    labels_list: np.ndarray,
    weights: np.ndarray = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: str = "avg",
    allows_overflow: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply weighted boxes fusion.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py

    Args:
        boxes_list: list of boxes predictions from each model, each box is 4 numbers.
        It has 3 dimensions (models_number, model_preds, 4)
        Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
        scores_list: list of scores for each model
        labels_list: list of labels for each model
        weights: list of weights for each model. Default: None, which means weight == 1 for each model
        iou_thr: IoU value for boxes to be a match
        skip_box_thr: exclude boxes with score lower than this variable
        conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value, 'box_and_model_avg': box and model wise hybrid weighted average, 'absent_model_aware_avg': weighted average that takes into account the absent model.
        allows_overflow: false if we want confidence score not exceed 1.0

    Returns:
        boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
        scores: confidence scores
        labels: boxes labels
    """
    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print(
            "Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.".format(
                len(weights), len(boxes_list)
            )
        )
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ["avg", "max", "box_and_model_avg", "absent_model_aware_avg"]:
        print(
            'Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(
                conf_type
            )
        )
        exit()

    filtered_boxes = prefilter_boxes(
        boxes_list, scores_list, labels_list, weights, skip_box_thr
    )
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes: List = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes: List = []
        weighted_boxes: List = []
        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())
        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = np.array(new_boxes[i])
            if conf_type == "box_and_model_avg":
                # weighted average for boxes
                weighted_boxes[i][1] = (
                    weighted_boxes[i][1] * len(clustered_boxes) / weighted_boxes[i][2]
                )
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i][1] = (
                    weighted_boxes[i][1] * clustered_boxes[idx, 2].sum() / weights.sum()
                )
            elif conf_type == "absent_model_aware_avg":
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i][1] = (
                    weighted_boxes[i][1]
                    * len(clustered_boxes)
                    / (weighted_boxes[i][2] + weights[mask].sum())
                )
            elif conf_type == "max":
                weighted_boxes[i][1] = weighted_boxes[i][1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i][1] = (
                    weighted_boxes[i][1]
                    * min(len(weights), len(clustered_boxes))
                    / weights.sum()
                )
            else:
                weighted_boxes[i][1] = (
                    weighted_boxes[i][1] * len(clustered_boxes) / weights.sum()
                )
        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes_np = np.concatenate(overall_boxes, axis=0)
    overall_boxes_np = overall_boxes_np[overall_boxes_np[:, 1].argsort()[::-1]]
    boxes = overall_boxes_np[:, 4:]
    scores = overall_boxes_np[:, 1]
    labels = overall_boxes_np[:, 0]
    return boxes, scores, labels


def prefilter_boxes_for_nmw(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    thr: float,
) -> Dict:
    """Filter bounding boxes, scores, and labels.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_nmw.py

    Args:
        boxes: list of boxes predictions from each model, each box is 4 numbers.
               It has 3 dimensions (models_number, model_preds, 4)
               Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
        scores: list of scores for each model
        labels: list of labels for each model
        weights: list of weights for each model. Default: None, which means weight == 1 for each model
        thr: threshold for boxes to keep (important for SoftNMS)

    Returns:
        new_boxes: filtered bounding boxes
    """
    # Create dict with boxes stored by its label
    new_boxes: Dict[int, List] = dict()
    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print(
                "Error. Length of boxes arrays not equal to length of scores array: {} != {}".format(
                    len(boxes[t]), len(scores[t])
                )
            )
            exit()

        if len(boxes[t]) != len(labels[t]):
            print(
                "Error. Length of boxes arrays not equal to length of labels array: {} != {}".format(
                    len(boxes[t]), len(labels[t])
                )
            )
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn("X2 < X1 value in box. Swap them.")
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn("Y2 < Y1 value in box. Swap them.")
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn("X1 < 0 in box. Set it to 0.")
                x1 = 0
            if x1 > 1:
                warnings.warn(
                    "X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range."
                )
                x1 = 1
            if x2 < 0:
                warnings.warn("X2 < 0 in box. Set it to 0.")
                x2 = 0
            if x2 > 1:
                warnings.warn(
                    "X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range."
                )
                x2 = 1
            if y1 < 0:
                warnings.warn("Y1 < 0 in box. Set it to 0.")
                y1 = 0
            if y1 > 1:
                warnings.warn(
                    "Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range."
                )
                y1 = 1
            if y2 < 0:
                warnings.warn("Y2 < 0 in box. Set it to 0.")
                y2 = 0
            if y2 > 1:
                warnings.warn(
                    "Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range."
                )
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            b = [int(label), float(score) * weights[t], x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box_for_nmw(boxes: List) -> np.ndarray:
    """Create weighted box for set of boxes.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_nmw.py

    Args:
        boxes: set of boxes to fuse

    Returns:
        box: weighted box
    """
    box = np.zeros(6, dtype=np.float32)
    best_box = boxes[0]
    conf = 0
    for b in boxes:
        iou = bb_intersection_over_union(b[2:], best_box[2:])
        weight = b[1] * iou
        box[2:] += weight * b[2:]
        conf += weight
    box[0] = best_box[0]
    box[1] = best_box[1]
    box[2:] /= conf
    return box


def find_matching_box_for_nmw(
    boxes_list: List, new_box: List, match_iou: float
) -> Tuple[int, float]:
    """Find matching bounding boxes.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_nmw.py

    Args:
        boxes_list: list of bounding boxes
        new_box: new bounding box
        match_iou: IoU threshold

    Returns:
        best_index: index of best matched bounding box
        best_iou: best IoU of best matched bounding box
    """
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def non_maximum_weighted(
    boxes_list: np.ndarray,
    scores_list: np.ndarray,
    labels_list: np.ndarray,
    weights: np.ndarray,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply non maximum weighted.

       Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_nmw.py

    Args:
        boxes_list: list of boxes predictions from each model, each box is 4 numbers.
        It has 3 dimensions (models_number, model_preds, 4)
        Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
        scores_list: list of scores for each model
        labels_list: list of labels for each model
        weights: list of weights for each model. Default: None, which means weight == 1 for each model
        iou_thr: IoU value for boxes to be a match
        skip_box_thr: exclude boxes with score lower than this variable

    Returns:
        boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
        scores: confidence scores
        labels: boxes labels
    """
    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print(
            "Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.".format(
                len(weights), len(boxes_list)
            )
        )
        weights = np.ones(len(boxes_list))
    weights = np.array(weights) / max(weights)
    # for i in range(len(weights)):
    #     scores_list[i] = (np.array(scores_list[i]) * weights[i])

    filtered_boxes = prefilter_boxes_for_nmw(
        boxes_list, scores_list, labels_list, weights, skip_box_thr
    )
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes: List = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes: List = []
        main_boxes: List = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box_for_nmw(main_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j].copy())
            else:
                new_boxes.append([boxes[j].copy()])
                main_boxes.append(boxes[j].copy())

        weighted_boxes = []
        for j in range(0, len(new_boxes)):
            box = get_weighted_box_for_nmw(new_boxes[j])
            weighted_boxes.append(box.copy())

        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes_np = np.concatenate(overall_boxes, axis=0)
    overall_boxes_np = overall_boxes_np[overall_boxes_np[:, 1].argsort()[::-1]]
    boxes = overall_boxes_np[:, 2:]
    scores = overall_boxes_np[:, 1]
    labels = overall_boxes_np[:, 0]
    return boxes, scores, labels
