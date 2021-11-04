"""Non Maximum Supression module.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

from typing import List

import torch
import torchvision

from scripts.utils.metrics import box_iou


def batched_nms(
    prediction: torch.Tensor,
    conf_thres: float = 0.001,
    iou_thres: float = 0.65,
    nms_box: int = 500,
    agnostic: bool = False,
    nms_type: str = "nms",
) -> List[torch.Tensor]:
    """Batched non max supression for faster run.

    Args:
        prediction: batch out (batch, n_obj, 4+1+n_class)
        conf_thres: confidence threshold.
        iou_thres: IoU threshold.
        nms_box: Number of boxes to use before check confidecne threshold.
        agnostic: Separate bboxes by classes for NMS with class separation.
        NMS type (e.g. nms, batched_nms, fast_nms, matrix_nms)

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
            box_offset = outputs[i][:, 5:6] * 4096  # Separate different classes.
            bboxes = outputs[i][:, :4] + box_offset
        else:
            bboxes = outputs[i][:, :4]

        # 1. torchvision nms (original)
        if nms_type == "nms":
            nms_idx = torchvision.ops.nms(bboxes, outputs[i][:, 4], iou_thres)
        # 2. torchvision batched_nms
        # https://github.com/ultralytics/yolov3/blob/f915bf175c02911a1f40fbd2de8494963d4e7914/utils/utils.py#L562-L563
        elif nms_type == "batched_nms":
            bboxes, scores = outputs[i][:, :4].clone(), outputs[i][:, 4]
            nms_idx = torchvision.ops.boxes.batched_nms(bboxes, outputs[i][:, 4], outputs[i][:, 5], iou_thres)
        # 3. fast nms (yolact)
        # https://github.com/ultralytics/yolov3/blob/77e6bdd3c1ea410b25c407fef1df1dab98f9c27b/utils/utils.py#L557-L559
        elif nms_type == "fast_nms":
            bboxes = outputs[i][:, :4].clone() + outputs[i][:, 5].view(-1, 1) * 4096
            iou = box_iou(bboxes, bboxes).triu_(diagonal=1)  # upper triangular iou matrix
            if len(iou) == 0: continue
            nms_idx = iou.max(0)[0] < iou_thres
        # 4. matrix nms
        # https://github.com/ultralytics/yolov3/issues/679#issuecomment-604164825
        elif nms_type == "matrix_nms":
            bboxes, scores = outputs[i][:, :4].clone(), outputs[i][:, 4]
            iou = box_iou(bboxes, bboxes).triu_(diagonal=1)  # upper triangular iou matrix
            if len(iou) == 0: continue
            m = iou.max(0)[0].view(-1, 1)  # max values
            decay = torch.exp(-(iou ** 2 - m ** 2) / 0.5).min(0)[0]  # gauss with sigma=0.5
            scores *= decay
            nms_idx = torch.full((bboxes.shape[0],), fill_value=1).bool()
        
        outputs[i] = outputs[i][nms_idx]

    return outputs
