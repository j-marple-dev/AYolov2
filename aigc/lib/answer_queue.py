"""Multiprocess queue module.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import abc
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import orjson
import torch
from lib.general_utils import scale_coords
from torch.multiprocessing import Process, Queue


class MultiProcessQueue(abc.ABC):
    """Abstract multiprocess queue."""

    def __init__(self) -> None:
        """Initialize queue."""
        self.queue: Queue = Queue()
        self.consumer_proc: Optional[Process] = None
        self.run = False

    def start(self) -> None:
        """Start producer-consumer queue."""
        if self.consumer_proc is not None:
            self.close()

        self.consumer_proc = Process(target=self._queue_proc)
        self.consumer_proc.daemon = True
        self.run = True
        self.consumer_proc.start()

    def add_queue(self, obj: Any) -> None:
        """Add queue from producer to consume from consumer.

        Args:
            obj: message to queue.
        """
        self.queue.put(obj)

    def _queue_proc(self) -> None:
        """Run while loop queue checker for the consumer."""
        while self.run:
            try:
                args = self.queue.get(timeout=0.1)
                self.consumer(args)
            except Empty:
                pass

    @abc.abstractmethod
    def consumer(self, args: Any) -> None:
        """Consume queue message.

        End event will be sent by sending 'DONE' message.

        Args:
            args: message to consume.
        """
        pass

    def close(self) -> None:
        """End consumer."""
        self.add_queue("DONE")
        if self.consumer_proc is not None:
            self.consumer_proc.join()


class ResultWriterBase(MultiProcessQueue, abc.ABC):
    """Abstract YOLO result writer queue processor."""

    """YOLO label id to COCO label id."""
    label_fixer = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]

    def __init__(self, file_name: str, n_param: int) -> None:
        """Initialize ResultWriter.

        Args:
            file_name: file name to write the result json file.
            n_param: Number of parameters of the model.
        """
        super().__init__()

        self.total_container: List[Dict] = [{"framework": "torch"}, {"Parameters": n_param}]
        self.seen_paths: Set[str] = set()
        self.file_name = file_name

    @abc.abstractmethod
    def scale_coords(self, *args: Any) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """Run scale coordinates.

        This method is called from self._add_outputs.

        Args:
            args: arguments to require in coordinates scaling.
        """
        pass

    def consumer(self, args: Any) -> None:
        """Consume message queue.

        Args:
            args: message queue.
                In this class, it's defined either
                (img_paths, yolo_outs, img_size, shapes)

                img_paths(List[str]): image path of each image.
                yolo_outs(List[torch.Tensor]): outputs from YOLO and NMS.
                img_size(Tuple[int, int]): batched image size (height, width)
                shapes(List[Tuple[Tuple[int, int], Tuple[Tuple[float, float], Tuple[int, int]]]])
                or str to end the consumer('DONE').
        """
        if isinstance(args, str) and args == "DONE":
            self.to_json(self.file_name)
            self.run = False
        else:
            self._add_outputs(*args)

    def add_outputs(
        self,
        names: List[str],
        outputs: List[torch.Tensor],
        img_size: Tuple[int, int],
        shapes: Optional[List[Tuple]] = None,
    ) -> None:
        """Add outputs to the queue.

        Args:
            img_paths: image path of each image.
            yolo_outs: outputs from YOLO and NMS.
            img_size: batched image size (height, width)
            shapes: List of shape information.
                    Each element contains,
                    (original image shape(height, width),
                    ((padding ratio(width, height),
                     (padding pixel(width, height)))))
        """
        outputs = [o.cpu().numpy() if o is not None else None for o in outputs]
        self.add_queue((names, outputs, img_size, shapes))

    def _add_outputs(
        self,
        names: List[str],
        outputs: List[torch.Tensor],
        img_size: Tuple[int, int],
        shapes: Optional[List[Tuple]] = None,
    ) -> None:
        """Add outputs (Run by consumer process).

        Args:
            img_paths: image path of each image.
            yolo_outs: outputs from YOLO and NMS.
            img_size: batched image size (height, width)
            shapes: List of shape information.
                    Each element contains,
                    (original image shape(height, width),
                    ((padding ratio(width, height),
                     (padding pixel(width, height)))))
        """
        for i in range(len(names)):
            if outputs[i] is None:
                scaled_bbox, conf = None, None
            else:
                bbox, conf = outputs[i][:, :4], outputs[i][:, 4:]
                if shapes is not None:
                    scaled_bbox = self.scale_coords(img_size, bbox, shapes[i])

                    if scaled_bbox is not None:
                        scaled_bbox[:, 2] = (
                            scaled_bbox[:, 2] - scaled_bbox[:, 0]
                        )  # [x1, y1, width, height]
                        scaled_bbox[:, 3] = scaled_bbox[:, 3] - scaled_bbox[:, 1]
                else:
                    scaled_bbox = bbox

            self.add_predicted_box(names[i], scaled_bbox, conf)

    def add_predicted_box(
        self,
        path: str,
        bboxes: Union[None, torch.Tensor, np.ndarray],
        confs: Union[None, torch.Tensor, np.ndarray],
    ) -> None:
        """Add predicted box.

        This method add predictions in single image.

        Args:
            path: image filepath. e.g.: "0608_V0011_000.jpg"
            predicts: predicted bboxes with shape [number_of_NMS_filtered_predictions, 4],
                whose row is [x1, y1, x2, y2].
            confs: confidence for bboxes with shape [number_of_NMS_filtered_predictions, ].
        """
        if path in self.seen_paths:
            return
        self.seen_paths.add(path)

        if bboxes is None or confs is None:
            objects = []
        else:
            objects = [
                {
                    "image_id": int(Path(path).stem.replace("t4_", "")),
                    "category_id": ResultWriterBase.label_fixer[int(conf[1])],
                    "bbox": [float(p) for p in row],
                    "score": float(conf[0]),
                }
                for row, conf in zip(bboxes, confs)
            ]

        self.total_container.extend(objects)

    def to_json(self, filepath: str) -> None:
        """Save result with json format.

        Args:
            filepath: filepath to save the result.
        """
        with open(filepath, "wb") as f:
            f.write(orjson.dumps(self.total_container))


class ResultWriterTorch(ResultWriterBase):
    """Result writer for PyTorch model."""

    def __init__(self, file_name: str, n_param: int) -> None:
        """Initialize ResultWriterTorch."""
        super().__init__(file_name, n_param)

    def scale_coords(  # type: ignore
        self,
        img_shape: Tuple[float, float],
        bboxes: Union[torch.Tensor, np.ndarray],
        ratio_pad: List,
    ) -> Optional[Union[torch.Tensor, np.ndarray]]:
        """Scale coordinates by the original image.

        Args:
            img_shape: current image shape. (h, w)
            bboxes: (xyxy) coordinates.
            ratio_pad: (original_image_size(h, w), and padding)

        Return:
            scaled coordinates.
        """
        if bboxes is None:
            return None

        # return scale_coords(img_shape, bboxes, ratio_pad[0], ratio_pad=ratio_pad[1])
        return scale_coords(
            img_shape, bboxes, ratio_pad[0]
        )  # This shows better result.
