"""Constants collection.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import inspect
import logging
import random
from typing import Any, Callable

import numpy as np

LOG_LEVEL = logging.DEBUG

"""Label names for datasets."""
LABELS = {
    "COCO": [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ],
    "VOC": [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ],
}

"""Plot color setting."""
PLOT_COLOR = tuple(
    map(
        tuple,
        np.concatenate(
            [
                (
                    np.vstack(
                        [
                            np.linspace(0, 2, 81) % 1,
                            np.linspace(0, 4, 81) % 1,
                            np.linspace(0, 6, 81) % 1,
                        ]
                    ).T[:-1]
                    * 150
                    + 50
                ).astype("int32")[i::10, :]
                for i in range(10)
            ]
        ),
    )
)

"""Unit test probability.
You can control global unit test probability by this variable.
"""
P_TEST = 0.5


def probably_run(p: float = P_TEST) -> Callable:
    """Run with probability (Decorator).

    Priority of probability to run is as follow.

    1. function(p=0.5)
    2. def function(p=0.5)
    3. @probably_run(p=0.5)
    4. @probably_run()  (Default p=0.5)

    The function does not necessarily to have p argument.
    You can simply define the function as

    def function():
        ...

    In this case, @probably_run(p=0.5) is used..

    Args:
        p: probability to run the function.
    """

    def decorator(function: Callable) -> Callable:
        """Wrap decorator function."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrap function."""
            prob = p
            default_p = inspect.signature(function).parameters.get("p", None)
            if default_p is not None:
                prob = default_p.default

            prob = kwargs.get("p", prob)

            if random.random() > prob:
                return

            return function(*args, **kwargs)

        return wrapper

    return decorator
