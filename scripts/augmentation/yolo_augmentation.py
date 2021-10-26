"""Augmentations in YOLOv5.

https://github.com/ultralytics/yolov5/blob/070af88108e5675358fd783aae9d91e927717322/utils/augmentations.py

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import math
import random
from typing import List, Tuple, Union

import cv2
import numpy as np

from scripts.utils.general import (box_candidates, resample_segments,
                                   segment2box)
from scripts.utils.metrics import bbox_ioa


def copy_paste(
    im: np.ndarray, labels: np.ndarray, segments: List[np.ndarray], p: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Copy-Paste augmentation https://arxiv.org/abs/2012.07177.

    Args:
        im: an image to apply copy-paste
        labels: bounding boxes (n, 5) array. (class_id, x1, y1, x2, y2)
        segments: segmentation array(Length of n). [(x1, y1, x2, y2, x3, y3, ...), ...]
        p: probability to apply copy-paste.
            Total number of copy-paste object = len(labels) * p

    Returns:
        Copy-pasted image,
        Copy-pasted bounding boxes,
        Copy-pasted segmentations,
    """
    n = len(segments)

    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(
                    im_new,
                    [segments[j].astype(np.int32)],
                    -1,
                    (255, 255, 255),
                    cv2.FILLED,
                )

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def get_center(bbox: np.ndarray) -> tuple:
    """Get centor of bbox.

    Args:
        bbox: bounding box.

    Returns:
        Centor coordinates of bbox.
    """
    return (bbox[3] - bbox[1]) / 2 + bbox[1], (bbox[4] - bbox[2]) / 2 + bbox[2]


def copy_paste2(
    im1: np.ndarray,
    labels1: np.ndarray,
    seg1: List[np.ndarray],
    im2: np.ndarray,
    labels2: np.ndarray,
    seg2: List[np.ndarray],
    scale_min: float,
    scale_max: float,
    p: float = 0.5,
    n_trial: int = 5,
    area_thr: float = 10,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Copy-paste augmentation in different images.a.

    Args:
        im1: base image.
        labels1: base image object labels.
        seg1: base image object segmentations.
        im2: source image.
        labels2: source image object labels.
        seg2: source image object segmentations.
        scale_min: scale factor min value.
        scale_max: scale factor max value.
        p: Probability of copy paste.

    Returns:
        Copy-pasted image,
        Copy-pasted bounding boxes,
        Copy-pasted segmentations,
    """
    n = len(seg2)
    if p and n:
        h, w, c = im1.shape  # height, width, channels
        im_new = np.zeros(im1.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels2[j], seg2[j]

            # move box coords and segmentation coords to 0, 0 start
            zero_moved_box = l - np.array([0, l[1], l[2], l[1], l[2]])
            zero_moved_seg = s - l[1:3]

            for _ in range(n_trial):
                # get scale factor
                scale_factor = random.uniform(scale_min, scale_max)

                # scale box with scale factors
                scaled_moved_box = zero_moved_box[1:] * scale_factor
                x = random.uniform(0, w - (scaled_moved_box[2] - scaled_moved_box[0]))
                y = random.uniform(0, h - (scaled_moved_box[3] - scaled_moved_box[1]))
                new_box = np.concatenate(([l[0]], scaled_moved_box)) + np.array(
                    [0, x, y, x, y]
                )
                ioa = bbox_ioa(new_box[1:5], labels1[:, 1:5])
                if (ioa < 0.3).all():
                    bbox_w = new_box[3] - new_box[1]
                    bbox_h = new_box[4] - new_box[2]

                    if bbox_w * bbox_h < area_thr or bbox_w == 0 or bbox_h == 0:
                        break
                    labels1 = np.concatenate((labels1, [new_box]), 0)
                    new_seg = zero_moved_seg * scale_factor + np.array([x, y])
                    seg1.append(new_seg)
                    img_temp = np.zeros(im2.shape, np.uint8)
                    cv2.drawContours(
                        img_temp,
                        [seg2[j].astype(np.int32)],
                        -1,
                        (255, 255, 255),
                        cv2.FILLED,
                    )
                    temp_result = cv2.bitwise_and(src1=im2, src2=img_temp)

                    # crop object
                    x1, y1, x2, y2 = int(l[1]), int(l[2]), int(l[3]), int(l[4])
                    temp_obj = temp_result[y1:y2, x1:x2, :]
                    obj = cv2.resize(temp_obj, (0, 0), fx=scale_factor, fy=scale_factor)

                    x1, y1, x2, y2 = (
                        int(x),
                        int(y),
                        int(x) + obj.shape[1],
                        int(y) + obj.shape[0],
                    )
                    im_new[y1:y2, x1:x2, :] = obj
                    break

        i = im_new > 0
        im1[i] = im_new[i]

    return im1, labels1, seg1


def random_perspective(
    im: np.ndarray,
    targets: np.ndarray,
    segments: Union[Tuple[np.ndarray, ...], List[np.ndarray]] = (),
    degrees: float = 10,
    translate: float = 0.1,
    scale: float = 0.1,
    shear: float = 10,
    perspective: float = 0.0,
    border: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment random perspective transform.

    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))

    Args:
        im: Image to apply augmentation
        targets: bounding boxes label (n, 5)
        segment: List of segmentations
        degrees: Rotate degree range (+/- degrees)
        translate: translation ratio (+/- translate)
        scale: scaling factor
        shear: shear transform factor
        perspective: perspective transform factor
        border: border to remove

    Returns:
        an augmented image,
        an augmented labels
    """
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(
                im, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            im = cv2.warpAffine(
                im, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = (
                    xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
                )  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(
                n, 8
            )  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = (
                np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            )

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(
            box1=targets[:, 1:5].T * s,
            box2=new.T,
            area_thr=0.01 if use_segments else 0.10,
        )
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def cutout(im: np.ndarray, labels: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Apply image cutout augmentation https://arxiv.org/abs/1708.04552.

    Args:
        im: an image to apply cutout (inplace)
        labels: bounding boxes (n, 5) array. (class_id, x1, y1, x2, y2)
        p: probability to apply copy-paste.
            Total number of copy-paste object = len(labels) * p

    Return:
        cutout applied label.
    """
    if random.random() >= p:
        return labels

    h, w = im.shape[:2]
    scales = (
        [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16
    )  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))  # create random masks
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(
    im: np.ndarray, labels: np.ndarray, im2: np.ndarray, labels2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf.

    Args:
        im: a first image to apply mixup
        labels: first bounding boxes (n, 5) array. (class_id, x1, y1, x2, y2)
        im2: a second image to apply mixup
        labels2: second bounding boxes (n, 5) array. (class_id, x1, y1, x2, y2)

    Returns:
        a mixup applied image,
        a mixup applied labels.
    """
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def augment_hsv(
    im: np.ndarray, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5
) -> None:
    """HSV color-space augmentation. (inplace).

    Args:
        im: image to apply hsv augmentation
        hgain: hue gain value
        sgain: saturation gain value
        vgain: value gain value
    """
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
        )
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
