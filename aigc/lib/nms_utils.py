"""Non Maximum Supression module.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

from typing import List

import torch
import torchvision


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
