"""Augmentation module.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import random
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np


class AugmentationPolicy:
    """Augmentation policy with albumentation."""

    def __init__(self, policy: Dict[str, Dict], prob: float = 1.0) -> None:
        """Augmentation with albumentation.

        policy: augmentation policy described in dictionary format.
            each key name represents albumentations.{KEY} augmentation and
            value contains keyword arguments for the albumentations.{KEY}
            EX) {"Blur": {"p": 0.5},
                 "Flip": {"p": 0.5}
                }
        prob: probability to run this augmentation policy
        """
        self.prob = prob

        self.transform = A.Compose(
            [
                getattr(__import__("albumentations", fromlist=[""]), aug_name)(**kwargs)
                for aug_name, kwargs in policy.items()
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    def __call__(
        self, img: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Augmentation function with label(optional).

        Args:
            img: image (HWC) to augment with (0 ~ 255) range.
            labels: (n, 5) labels. (class_id, x1, y1, x2, y2) with pixel coordinates.

        Returns:
            augmented image if labels is None.
            (augmented image, labels) if labels is not None.
        """
        aug_labels = np.array(((0, 0.1, 0.1, 0.1, 0.1),)) if labels is None else labels

        if random.random() < self.prob:
            augmented = self.transform(
                image=img, bboxes=aug_labels[:, 1:], class_labels=aug_labels[:, 0]
            )
            im = augmented["image"]
            aug_labels = np.hstack(
                [
                    np.array(augmented["class_labels"]).reshape(-1, 1),
                    np.array(augmented["bboxes"]),
                ]
            )
        else:
            im = img

        if labels is not None:
            return im, aug_labels
        else:
            return im


class MultiAugmentationPolicies:
    """Multiple augmentation policies with albumentations."""

    def __init__(self, policies: List[Dict]) -> None:
        """Multiple augmentation with albumentation.

        policies: List of augmentation policies described in dictionary format.
            each key name represents albumentations.{KEY} augmentation and
            value contains keyword arguments for the albumentations.{KEY}
            EX) [
                    {
                    "policy":
                        {
                            "Blur": {"p": 0.5},
                             "Flip": {"p": 0.5}
                        },
                    "prob": 0.3
                    },
                    {
                    "policy":
                        {
                            "RandomGamma": {"p": 0.5},
                             "HorizontalFlip": {"p": 0.5}
                        },
                    "prob": 0.3
                    }
                ]
        """
        self.transforms = [
            AugmentationPolicy(aug["policy"], aug["prob"]) for aug in policies
        ]

    def __call__(
        self, img: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply multiple augmentation policy with label(optional).

        Args:
            img: image (HWC) to augment with (0 ~ 255) range.
            labels: (n, 5) labels. (class_id, x1, y1, x2, y2) with pixel coordinates.

        Returns:
            augmented image if labels is None.
            (augmented image, labels) if labels is not None.
        """
        for transform in self.transforms:
            if labels is not None:
                img, labels = transform(img, labels)
            else:
                img = transform(img)  # type: ignore

        return img, labels if labels is not None else img
