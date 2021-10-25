"""Unit test Model Converter.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import gc
import importlib
import os
import random
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from kindle import Model, YOLOModel

from scripts.model_converter.model_converter import ModelConverter


def test_model_converter_torchscript(p: float = 0.5) -> None:
    if random.random() > p:
        return

    test_input = torch.rand((8, 3, 320, 320))
    model = YOLOModel(
        os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=True
    )
    model.eval()
    out_tensor = model(test_input)
    model.export(verbose=True)
    out_tensor_export = model(test_input)

    converter = ModelConverter(model, 8, (320, 320), verbose=2)
    converter.dry_run()
    converter.to_torch_script(os.path.join("tests", ".model.ts"))
    ts_model = torch.jit.load(os.path.join("tests", ".model.ts"))
    out_tensor_ts = ts_model(test_input)

    assert torch.isclose(out_tensor[0], out_tensor_export[0]).all()
    assert torch.isclose(out_tensor_export[0], out_tensor_ts[0]).all()

    os.remove(os.path.join("tests", ".model.ts"))

    del test_input, model, out_tensor, converter, ts_model, out_tensor_ts
    gc.collect()


def test_model_converter_onnx(p: float = 0.5) -> None:
    if random.random() > p:
        return

    test_input = torch.rand((8, 3, 320, 320))
    model = YOLOModel(
        os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=True
    )
    model.eval()
    out_tensor = model(test_input)
    model.export(verbose=True)
    out_tensor_export = model(test_input)

    converter = ModelConverter(model, 8, (320, 320), verbose=2)
    converter.dry_run()
    converter.to_onnx(os.path.join("tests", ".model.onnx"))
    onnx_model = ort.InferenceSession(os.path.join("tests", ".model.onnx"))

    out_tensor_onnx = onnx_model.run(
        input_feed={"images": test_input.numpy()}, output_names=["output"]
    )

    assert torch.isclose(out_tensor[0], out_tensor_export[0]).all()
    assert np.isclose(out_tensor_export[0].detach().numpy(), out_tensor_onnx[0]).all()

    os.remove(os.path.join("tests", ".model.onnx"))

    del test_input, model, out_tensor, out_tensor_export, converter, out_tensor_onnx
    gc.collect()


def test_model_converter_tensorrt(
    keep_trt: bool = False, check_trt_exists: bool = False
) -> None:
    test_input = torch.rand((8, 3, 640, 640))
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. Skip test_model_converter_tensorrt")
        return
    if importlib.util.find_spec("tensorrt") is None:
        print("TensorRT is not installed. Skip test_model_converter_tensorrt")
        return

    import tensorrt as trt

    from scripts.utils.tensorrt_runner import TrtWrapper

    model = YOLOModel(
        os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=True
    ).to(torch.device("cuda:0"))

    model.eval().export()
    torch_out = model(test_input.to(torch.device("cuda:0")))

    profiler = model.profile(
        verbose=True, n_run=100, input_size=(640, 640), batch_size=8
    )

    model = (
        YOLOModel(
            os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=False
        )
        .eval()
        .export()
    )

    converter = ModelConverter(model, 8, (640, 640), verbose=2)
    converter.dry_run()

    if not (check_trt_exists and os.path.isfile(os.path.join("tests", ".model.trt"))):
        convert_ok = converter.to_tensorrt(
            os.path.join("tests", ".model.trt"),
            fp16=True,
            opset_version=11,
            build_nms=False,
        )
        assert convert_ok

    trt_model = TrtWrapper(
        os.path.join("tests", ".model.trt"), 8, torch.device("cuda:0")
    )
    test_input = test_input.to(torch.device("cuda:0"), non_blocking=True).contiguous()
    t0 = time.monotonic()
    for _ in range(100):
        trt_out = trt_model(test_input, raw_out=True)[0]
    t_trt = time.monotonic() - t0

    print(
        f"Torch model time: {profiler.total_run_time:.2f}s, TensorRT model time: {t_trt:.2f}s"
    )
    if not keep_trt:
        os.remove(os.path.join("tests", ".model.trt"))

    # Bounding boxes
    assert torch.isclose(torch_out[0][:, :, :4], trt_out[:, :, :4], rtol=0.2).all()
    # Object and class scores
    assert torch.isclose(torch_out[0][:, :, 4:], trt_out[:, :, 4:], rtol=0.1).all()


if __name__ == "__main__":
    test_model_converter_torchscript()
    test_model_converter_onnx()
    # test_model_converter_tensorrt(keep_trt=True, check_trt_exists=False)
