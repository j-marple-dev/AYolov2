"""Unit test Model Converter.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import importlib
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from kindle import Model, YOLOModel

from scripts.model_converter.model_converter import ModelConverter


def test_model_converter_torchscript() -> None:
    test_input = torch.rand((8, 3, 640, 640))
    model = YOLOModel(
        os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=True
    )
    model.eval()
    out_tensor = model(test_input)
    model.export(verbose=True)
    out_tensor_export = model(test_input)

    converter = ModelConverter(model, 8, (640, 640), verbose=2)
    converter.dry_run()
    converter.to_torch_script(os.path.join("tests", ".model.ts"))
    ts_model = torch.jit.load(os.path.join("tests", ".model.ts"))
    out_tensor_ts = ts_model(test_input)

    assert torch.isclose(out_tensor[0], out_tensor_export[0]).all()
    assert torch.isclose(out_tensor_export[0], out_tensor_ts[0]).all()

    os.remove(os.path.join("tests", ".model.ts"))


def test_model_converter_onnx() -> None:
    test_input = torch.rand((8, 3, 640, 640))
    model = YOLOModel(
        os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=True
    )
    model.eval()
    out_tensor = model(test_input)
    model.export(verbose=True)
    out_tensor_export = model(test_input)

    converter = ModelConverter(model, 8, (640, 640), verbose=2)
    converter.dry_run()
    converter.to_onnx(os.path.join("tests", ".model.onnx"))
    onnx_model = ort.InferenceSession(os.path.join("tests", ".model.onnx"))

    out_tensor_onnx = onnx_model.run(
        input_feed={"images": test_input.numpy()}, output_names=["output"]
    )

    assert torch.isclose(out_tensor[0], out_tensor_export[0]).all()
    assert np.isclose(out_tensor_export[0].detach().numpy(), out_tensor_onnx[0]).all()

    os.remove(os.path.join("tests", ".model.onnx"))


def test_model_converter_tensorrt() -> None:
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. Skip test_model_converter_tensorrt")
        return
    if importlib.util.find_spec("tensorrt") is None:
        print("TensorRT is not installed. Skip test_model_converter_tensorrt")
        return

    model = YOLOModel(
        os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=True
    )

    model.eval().export()
    profiler = model.profile(verbose=True, n_run=1, input_size=(640, 640), batch_size=8)

    converter = ModelConverter(model, 8, (640, 640), verbose=2)
    converter.dry_run()

    converter.to_torch_script(os.path.join("tests", ".model.ts"))
    # converter.to_onnx(os.path.join("tests", ".model.onnx"))
    converter.to_tensorrt(
        os.path.join("tests", ".model.trt"), fp16=True, opset_version=11
    )


if __name__ == "__main__":
    # test_model_converter_torchscript()
    # test_model_converter_onnx()
    test_model_converter_tensorrt()
