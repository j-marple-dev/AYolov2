"""Multiprocess queue module.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import abc
import json
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.multiprocessing import Process, Queue

from scripts.utils.general import scale_coords, xyxy2xywh


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

    def __init__(self, file_name: str) -> None:
        """Initialize ResultWriter.

        Args:
            file_name: file name to write the result json file.
        """
        super().__init__()

        self.total_container: List[Dict] = []
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
            bbox = outputs[i][:, :4] if outputs[i] is not None else None
            shape = None if shapes is None else shapes[i]
            scaled_bbox = self.scale_coords(img_size, bbox, shape)

            # Normalize and xyxy to xywh
            if shape is not None:
                scaled_bbox = xyxy2xywh(scaled_bbox, wh=shape[0][::-1])

            conf = outputs[i][:, 4:] if outputs[i] is not None else None
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

        # TODO(jeikeilim): Make xyxy to xywh option.
        if bboxes is None or confs is None:
            objects = []
        else:
            objects = [
                {
                    "image_id": int(Path(path).stem),
                    "category_id": int(conf[1]),
                    "bbox": [float(p) for p in row],
                    "score": float(conf[0]),
                }
                for row, conf in zip(bboxes, confs)
            ]

        self.total_container.extend(objects)

    # def filter_small_box(self):
    #     for i, annot in enumerate(self.total_container['annotations']):
    #         obj_candidate = []
    #         for j, obj_annot in enumerate(annot['objects']):
    #             pos = np.array(obj_annot['position'])
    #             w = np.diff(pos[0::2])
    #             h = np.diff(pos[1::2])
    #             if w >= 32 and h >= 32:
    #                 obj_candidate.append(obj_annot)

    #         self.total_container['annotations'][i]['objects'] = obj_candidate

    def to_json(self, filepath: str) -> None:
        """Save result with json format.

        Args:
            filepath: filepath to save the result.
        """
        # self.filter_small_box()
        with open(filepath, "w") as f:
            json.dump(self.total_container, f)


class ResultWriterTorch(ResultWriterBase):
    """Result writer for PyTorch model."""

    def __init__(self, file_name: str) -> None:
        """Initialize ResultWriterTorch."""
        super().__init__(file_name)

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

        return scale_coords(img_shape, bboxes, ratio_pad[0], ratio_pad=ratio_pad[1])
