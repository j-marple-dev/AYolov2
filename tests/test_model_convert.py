"""Unit test Model Converter.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import os
import torch

from kindle import YOLOModel
from scripts.model_converter.model_converter import ModelConverter


def test_model_converter() -> None:
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = YOLOModel(
        os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=True
    )
    # model.eval().fuse()
    model.model[-1].export = True

    converter = ModelConverter(model, 8, (640, 640), verbose=2)
    converter.dry_run()

    # converter.to_torch_script(os.path.join("tests", ".model.ts"))
    # converter.to_onnx(os.path.join("tests", ".model.onnx"))
    converter.to_tensorrt(os.path.join("tests", ".model.trt"), fp16=True)


if __name__ == "__main__":
    test_model_converter()

