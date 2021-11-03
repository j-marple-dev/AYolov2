"""Evaluation metric modules.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tabulate import tabulate
from tqdm import tqdm

from scripts.utils.general import xywh2xyxy
from scripts.utils.logger import get_logger
from scripts.utils.plot_utils import draw_labels, plot_mc_curve, plot_pr_curve

LOGGER = get_logger(__name__)


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute the intersection over box2 area given box1, box2.

    Boxes are x1y1x2y2

    Args:
        box1: shape (4)
        box2: shape (n, 4)

    Return:
        shape (n)
    """
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
        np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
    ).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    x1y1x2y2: bool = True,
    g_iou: bool = False,
    d_iou: bool = False,
    c_iou: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute bounding boxes IOU.

    Args:
        box1: first bounding boxes. (4, n)
        box2: first bounding boxes. (n, 4)
        x1y1x2y2: True if coordinates are xyxy format.
        g_iou: compute GIoU value
        d_iou: compute DIoU value
        c_iou: compute CIoU value
        eps: epsilon value for numerical stabilization

    Returns:
        the IoU of box1 to box2.
        box1 is 4, box2 is nx4
    """
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if g_iou or d_iou or c_iou:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if (
            c_iou or d_iou
        ):  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if d_iou:
                return iou - rho2 / c2  # d_iou
            elif (
                c_iou
            ):  # https://github.com/Zzh-tju/d_iou-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # c_iou
        else:  # g_iou https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # g_iou

    return iou  # IoU


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute intersection of union (Jaccard index) of boxes.

    Args:
        box1: a torch tensor with (N, 4).
        box2: a torch tensor with (N, 4).

    Returns:
        iou: (N, M) torch tensor. the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2.
    """

    def box_area(box: torch.Tensor) -> torch.Tensor:
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (area1[:, None] + area2 - inter)


class ConfusionMatrix:
    """Updated version of OD confusion matrix.

    https://github.com/kaanakan/object_detection_confusion_matrix.
    """

    def __init__(self, nc: int, conf: float = 0.25, iou_thres: float = 0.45) -> None:
        """Initialize ConfusionMatrix class.

        Args:
            nc: number of classes.
            conf: confidence threshold.
            iou_thres: IoU threshold.
        """
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections: np.ndarray, labels: np.ndarray) -> None:
        """Return intersection-over-union (Jaccard index) of boxes.

        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

        Args:
            detections: (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels: (Array[M, 5]), class, x1, y1, x2, y2

        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .detach()
                .cpu()
                .numpy()
            )
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def get_matrix(self) -> np.ndarray:
        """Return matrix."""
        return self.matrix

    def plot(self, names: list, normalize: bool = True, save_dir: str = "",) -> None:
        """Plot confusion matrix.

        Args:
            names: class names with order.
            normalize: Normalize flag.
            save_dir: directory where the plot images will be saved.
        """
        try:
            import seaborn as sn

            array = self.matrix / (
                (self.matrix.sum(0).reshape(1, -1) + 1e-6) if normalize else 1
            )  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(
                names
            ) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(
                    array,
                    annot=self.nc < 30,
                    annot_kws={"size": 8},
                    cmap="Blues",
                    fmt=".2f",
                    square=True,
                    xticklabels=names + ["background FP"] if labels else "auto",
                    yticklabels=names + ["background FN"] if labels else "auto",
                ).set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel("True")
            fig.axes[0].set_ylabel("Predicted")
            fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
            plt.close()
        except Exception as e:
            LOGGER.warn(f"WARNING: ConfusionMatrix plot failure: {e}")

    def print(self) -> None:
        """Print confusion matrix."""
        for i in range(self.nc + 1):
            LOGGER.info(" ".join(map(str, self.matrix[i])))


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[list] = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels: Union[tuple, list] = (),
    max_det: int = 300,
) -> list:
    """Run Non-Maximum Suppression (NMS) on inference results.

    Args:
        prediction: model output.
        conf_thres: confidence threshold.
        iou_thres: IoU threshold.
        classes: Debug purpose to save both ground truth label and predicted result.
        agnostic: Separate bboxes by classes for NMS with class separation.
        multi_label: multiple labels per box.
        labels: labels.
        max_det: maximum number of detected objects by model.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh = 2  # noqa
    max_wh = 4096
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            label = labels[xi]
            v = torch.zeros((len(label), nc + 5), device=x.device)
            v[:, :4] = label[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(label)), label[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warn(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output


def compute_ap(
    recall: list, precision: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the average precision, given the recall and precision curves.

    Args:
        recall:    The recall curve (list)
        precision: The precision curve (list)
    Returns:
        Average precision, precision curve, recall curve
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    save_dir: str = ".",
    names: list = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the average precision, given the recall and precision curves.

    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

    Args:
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory

    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(
                -px, -conf[i], recall[:, 0], left=0
            )  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / "PR_curve.png", names)
        plot_mc_curve(px, f1, Path(save_dir) / "F1_curve.png", names, ylabel="F1")
        plot_mc_curve(px, p, Path(save_dir) / "P_curve.png", names, ylabel="Precision")
        plot_mc_curve(px, r, Path(save_dir) / "R_curve.png", names, ylabel="Recall")

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype("int32")


def check_correct_prediction_by_iou(
    detections: torch.Tensor,
    labels: torch.Tensor,
    iou_s: float = 0.5,
    iou_e: float = 0.95,
    iou_step: float = 0.05,
) -> torch.Tensor:
    """Return correct predictions of matrix by IoU.

    Both sets of boxes are in (x1, y1, x2, y2) format.

    Args:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
        iou_s: start of IoU check value.
        iou_e: end of IoU check value.
        iou_step: step of IoU check value.
            i.e. (iou_s, iou_e, iou_step) = (0.5, 0.95, 0.05)
                --> torch.arange(iou_s, iou_e + iou_step, iou_step)
                    --> (0.5, 0.55, 0.60, 0.65, ..., 0.85, 0.90, 0.95)

    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    device = detections.device

    iouv = torch.arange(iou_s, iou_e + iou_step, iou_step, device=device)

    iou = box_iou(labels[:, 1:], detections[:, :4])

    correct = torch.zeros(
        (detections.shape[0], iouv.shape[0]), dtype=torch.bool, device=device
    )
    matched = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if matched[0].shape[0] > 0:
        matches = (
            torch.cat(
                (torch.stack(matched, 1), iou[matched[0], matched[1]][:, None]), 1
            )
            .cpu()
            .numpy()
        )
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

        matches = torch.Tensor(matches, device=device)

        correct[matches[:, 1].long()] = matches[:, 2:3] > iouv

    return correct


class COCOmAPEvaluator:
    """COCO mAP evaluator with json file."""

    def __init__(
        self,
        gt_path: str,
        img_root: Optional[str] = None,
        export_root: Optional[str] = None,
    ) -> None:
        """Initialize COCOmAPEvaluator.

        Args:
            gt_path: ground truth annotation path.
                (Usually, a path of 'instances_val2017.json')
            img_root: image root directory for ground truth annotation.
                (This is only necessary for debug purpose.)
            export_root: export inference result root directory.
                (This is only necessary for debug purpose.)
        """
        with open(gt_path, "r") as f:
            self.gt_labels = json.load(f)

        self.fix_label = {
            category["id"]: i for i, category in enumerate(self.gt_labels["categories"])
        }
        self.names = [category["name"] for category in self.gt_labels["categories"]]
        self.unique_img_id = set(
            [annot["image_id"] for annot in self.gt_labels["annotations"]]
        )
        self.gt: Dict[int, List] = {img_id: [] for img_id in self.unique_img_id}

        for gt_label in self.gt_labels["annotations"]:
            self.gt[gt_label["image_id"]].append(
                {k: gt_label[k] for k in ["image_id", "bbox", "category_id"]}
            )

        self.img_sizes = {
            gt_label["id"]: (gt_label["height"], gt_label["width"])
            for gt_label in self.gt_labels["images"]
        }
        self.img_root = img_root
        self.export_root = export_root

        if self.export_root is not None:
            os.makedirs(self.export_root, exist_ok=True)

    def evaluate(self, label_path: str, debug: bool = False) -> Dict[str, Any]:
        """Evaulate mAP value from JSON file.

        Args:
            label_path: label .json file path. The JSON file should be as below.
                [
                    {
                        "image_id": IMAGE_ID(int),
                        "category_id": CATEGORY_ID(int),
                        "bbox": [CX, CY, WIDTH, HEIGHT],
                        "score": CONFIDENCE
                    },
                    ...
                ]
            debug: debug flag to plot or save the inference result.
                If self.img_root and self.export_root is given,
                the inference result will be saved.
                Otherwise it will be plotted.

        Return:
            Dictionary of the result.

            {
                "p": precision,
                "r": recall,
                "ap": ap,
                "ap50": ap50,
                "f1": f1,
                "mp": mp,
                "mr": mr,
                "map50": map50,
                "map50_95": map50_95,
                "target_histogram": target_histogram,
                "names": self.names,
            }
        """
        with open(label_path, "r") as f:
            preds = json.load(f)

        corrects = []
        unique_img_id = set([pred["image_id"] for pred in preds])
        labels: Dict[int, List] = {img_id: [] for img_id in unique_img_id}
        for pred in preds:
            labels[pred["image_id"]].append(pred)

        img_ids = set(list(self.unique_img_id) + list(unique_img_id))
        # img_ids = unique_img_id

        confusion_matrix = ConfusionMatrix(nc=len(self.names))

        for img_id in tqdm(img_ids, "Compute score ..."):
            if img_id not in unique_img_id:
                label_pred = torch.zeros(0, 6)
            else:
                label_pred = torch.tensor(
                    [
                        [*label["bbox"], label["score"], label["category_id"]]
                        for label in labels[img_id]
                    ]
                )
                label_pred[:, :4] = xywh2xyxy(label_pred[:, :4])

            if img_id not in self.unique_img_id:
                label_gt = torch.zeros(0, 5)
            else:
                label_gt = torch.tensor(
                    [
                        [self.fix_label[label["category_id"]], *label["bbox"]]
                        for label in self.gt[img_id]
                    ]
                )

                label_gt[:, 3:5] += label_gt[:, 1:3]  # COCO xy is at top-left point.

            correct = check_correct_prediction_by_iou(label_pred, label_gt)
            corrects.append(
                (correct, label_pred[:, 4], label_pred[:, 5], label_gt[:, 0])
            )

            confusion_matrix.process_batch(label_pred, label_gt)  # type: ignore

            if debug:
                self._draw_result(img_id, label_pred, label_gt)

        if self.export_root is not None:
            confusion_matrix.plot(self.names, save_dir=self.export_root)

        corrects_np = [np.concatenate(x, 0) for x in zip(*corrects)]
        precision, recall, ap, f1, ap_class = ap_per_class(
            corrects_np[0],
            corrects_np[1],
            corrects_np[2],
            corrects_np[3],
            plot=(self.export_root is not None),
            save_dir=(self.export_root if self.export_root is not None else ""),
            names=self.names,
        )
        ap50, ap = ap[:, 0], ap.mean(1)  # type: ignore
        mp, mr, map50, map50_95 = (
            precision.mean(),
            recall.mean(),
            ap50.mean(),
            ap.mean(),
        )
        target_histogram = np.bincount(
            corrects_np[3].astype(np.int64), minlength=len(self.names)
        )

        result = {
            "p": precision,
            "r": recall,
            "ap": ap,
            "ap50": ap50,
            "f1": f1,
            "mp": mp,
            "mr": mr,
            "map50": map50,
            "map50_95": map50_95,
            "target_histogram": target_histogram,
            "names": self.names,
        }

        self.print_result(result)

        return result

    @staticmethod
    def print_result(result: Dict) -> None:
        """Print result dictionary with tabulate.

        Args:
            result: result dictionary generated by self.evaluate
        """
        result_by_class = np.stack(
            (
                result["target_histogram"],
                result["p"],
                result["r"],
                result["f1"],
                result["ap50"],
                result["ap"],
            ),
            1,
        )
        result_by_all = np.array(
            [
                result["target_histogram"].sum(),
                result["mp"],
                result["mr"],
                result["f1"].mean(),
                result["map50"],
                result["map50_95"],
            ]
        )
        names = np.array(result["names"] + ["all"])

        contents = np.concatenate(
            (names[:, None], np.vstack((result_by_class, result_by_all))), 1
        )
        LOGGER.info(
            "\n"
            + tabulate(
                contents,
                headers=["name", "n_targets", "P", "R", "F1", "mAP50", "mAP50:95"],
                tablefmt="github",
            )
        )

    def _draw_result(
        self, img_id: int, label_pred: torch.Tensor, label_gt: torch.Tensor
    ) -> None:
        """Draw or save inference result.

        Args:
            img_id: image id for label and ground truth.
            label_pred: prediction result (n, 6)
            label_gt: ground truth label (n, 5)
        """
        if self.img_root is None:
            return

        img_name = f"{img_id:012d}.jpg"
        img_path = os.path.join(self.img_root, img_name)

        if not os.path.isfile(img_path):
            return

        img = cv2.imread(img_path)

        img_pred = draw_labels(
            img.copy(),
            np.concatenate((label_pred[:, 5:6], label_pred[:, :4]), 1),
            {i: self.names[i] for i in range(len(self.names))},
            norm_xywh=False,
        )
        img_gt = draw_labels(
            img.copy(),
            np.concatenate((label_gt[:, 0:1], label_gt[:, 1:]), 1),
            {i: self.names[i] for i in range(len(self.names))},
            norm_xywh=False,
        )

        img_merge = np.concatenate(
            (
                img_pred,
                np.full(
                    (img_gt.shape[0], int(img_gt.shape[1] * 0.03), 3),
                    127,
                    dtype=np.uint8,
                ),
                img_gt,
            ),
            1,
        )

        if self.export_root is not None:
            if self.export_root == self.img_root:
                LOGGER.warn(
                    "Export and image root directory is same! Skip saving the result."
                )
            else:
                export_path = os.path.join(self.export_root, img_name)
                cv2.imwrite(export_path, img_merge)
        else:
            cv2.imshow("result", img_merge)
            cv2.waitKey(0)
