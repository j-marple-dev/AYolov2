"""Model converter module.

Converts PyTorch model into TorchScript, ONNX and, TensorRT.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import os
from typing import Any, Optional, Tuple, Union

import onnx
from onnxsim import simplify
import torch
from torch import nn
from scripts.utils.activations import SiLU, Hardswish

# from src.tensorrt.int8_calibrator import Int8Calibrator


def simplify_onnx(onnx_path: str) -> None:
    """Simplify ONNX model.
    Args:
        onnx_path: ONNX model path to simplify.
    """
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)


class ModelConverter:
    """PyTorch model converter class."""

    TRT_MAX_BATCH_SIZE: int = 1

    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 32,
        img_size: Union[int, Tuple[int, int]] = 32,
        verbose: int = 1,
    ) -> None:
        """Initialize ModelConverter instance.
        Args:
            model: PyTorch module.
            batch_size: batch size is required when converting to TensorRT model.
                        Otherwise, it will be ignored.
            img_size: input size of the model. Batch size and channel is excluded.
            verbose: verbosity level
        """
        if isinstance(img_size, int):
            input_size = (img_size, img_size)
        else:
            input_size = img_size

        self.verbose = verbose
        self.model = model
        self.batch_size = batch_size
        self.device = next(self.model.parameters()).device
        self.test_input = torch.zeros(
            batch_size, 3, *input_size, device=self.device
        )

    def dry_run(self) -> None:
        """Dry run the model for memory load purpose."""
        self.model.eval()
        self.model(self.test_input)

    def to_torch_script(self, path: str) -> None:
        """Export model to TorchScript.
        Args:
            path: export path.
        """
        ts = torch.jit.trace(self.model, self.test_input)
        ts.save(path)
        # I might need this?
        # zipf = f
        # try:
        #     tempname = os.path.join(tempdir, 'test.zip')
        #     with zipfile.ZipFile(zipf, 'r') as zipread:
        #         with zipfile.ZipFile(tempname, 'w') as zipwrite:
        #             for item in zipread.infolist():
        #                 data = zipread.read(item.filename)
        #                 if 'yolo.py' in item.filename:
        #                     data = data.replace(b"cpu", b"cuda:0")
        #                 zipwrite.writestr(item, data)

    def to_onnx(self, path: str, opset_version: int = 11, **kwargs) -> None:
        """Export model to ONNX model.
        Args:
           path: export path
           opset_version: ONNX op set version.
        """
        # temp convert model
        for k, m in self.model.named_modules():
            if hasattr(m, "activation"):
                if isinstance(m.activation, nn.Hardswish):
                    m.activation = Hardswish()
                elif isinstance(m.activation, nn.SiLU):
                    m.activation = SiLU()

        torch.onnx.export(
            self.model,
            self.test_input,
            path,
            verbose=self.verbose > 1,
            opset_version=opset_version,
            input_names=["images"],
            output_names=["output"],
            **kwargs,
        )

        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        try:
            simplify_onnx(path)
        except Exception as e:
            print("Simplifying ONNX has failed. However, ")
            print("ONNX model still has been converted successfully.")
            print(f"Reason: {e}")

    def to_tensorrt(  # pylint: disable=too-many-branches, too-many-statements
        self,
        path: str,
        opset_version: int = 11,
        fp16: bool = False,
        int8: bool = False,
        dla_core_id: int = -1,
        workspace_size_gib: int = 1,
        int8_calibrator: Optional[Any] = None,
    ) -> None:
        """Convert model to TensorRT model.
        Args:
            path: export path
            opset_version: ONNX op set version. Recommended to stick with 11.
            int8: convert model to INT8 precision. int8_calibrator must be provided.
            int8_calibrator: int8 calibrator instance. (tensorrt.IInt8EntropyCalibrator2)
        """
        try:
            import tensorrt as trt  # pylint: disable=import-outside-toplevel, import-error
        except ModuleNotFoundError as error:
            raise Exception("TensorRT is not installed.") from error

        onnx_path = f"{path}.onnx"
        self.to_onnx(onnx_path, opset_version=opset_version)

        if not os.path.isfile(onnx_path):
            return
        with open(onnx_path, "rb") as f:
            onnx_data = f.read()

        if self.verbose > 0:
            trt_logger = trt.Logger(trt.Logger.VERBOSE)
        else:
            trt_logger = trt.Logger()

        if trt.__version__[0] < "7":
            explicit_batch = []
        else:
            explicit_batch = [
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            ]

        with trt.Builder(trt_logger) as builder, builder.create_network(
            *explicit_batch
        ) as network, trt.OnnxParser(network, trt_logger) as parser:
            if int8 and not builder.platform_has_fast_int8:
                print(
                    "INT8 is not supported by this platform. Switching to float precision."
                )
                int8 = False

            if int8 and int8_calibrator is None:
                print("INT8 calibrator must be provided. Switching to float precision.")
                int8 = False

            import pdb; pdb.set_trace()
            if not parser.parse(onnx_data):
                print("Failed to parse the ONNX file.")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return

            if trt.__version__[0] >= "7":
                network.get_input(0).shape = (self.batch_size,) + network.get_input(
                    0
                ).shape[1:]
            # Add custom layer here if deisred.

            builder.max_batch_size = self.TRT_MAX_BATCH_SIZE
            # builder.max_batch_size = self.batch_size
            if trt.__version__[0] < "7":
                if dla_core_id >= 0:
                    raise RuntimeError(
                        "DLA core is not supported by older version API. "
                        f"Current TensorRT version: {trt.__version__}"
                    )
                builder.max_workspace_size = workspace_size_gib << 30
                builder.fp16_mode = builder.platform_has_fast_fp16 if fp16 else False
                if int8:
                    builder.int8_mode = True
                    builder.int8_calibrator = int8_calibrator
                engine = builder.build_cuda_engine(network)
            else:
                config = builder.create_builder_config()
                config.max_workspace_size = workspace_size_gib << 30
                config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
                if fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                profile = builder.create_optimization_profile()
                profile.set_shape(
                    "images",
                    (
                        # self.TRT_MAX_BATCH_SIZE,
                        self.batch_size,
                        3,
                        self.test_input.shape[2],
                        self.test_input.shape[3],
                    ),
                    (
                        # self.TRT_MAX_BATCH_SIZE,
                        self.batch_size,
                        3,
                        self.test_input.shape[2],
                        self.test_input.shape[3],
                    ),
                    (
                        # self.TRT_MAX_BATCH_SIZE,
                        self.batch_size,
                        3,
                        self.test_input.shape[2],
                        self.test_input.shape[3],
                    ),
                )
                config.add_optimization_profile(profile)

                if int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    config.int8_calibrator = int8_calibrator
                    config.set_calibration_profile(profile)
                if dla_core_id >= 0:
                    config.default_device_type = trt.DeviceType.DLA
                    config.DLA_core = dla_core_id
                    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                    print(f"Using DLA Core id {dla_core_id}")
                engine = builder.build_engine(network, config)

        os.remove(onnx_path)

        if engine is not None:
            with open(path, "wb") as f:
                f.write(engine.serialize())
            print(f"Completed converting to TensorRT model file to {path}")
        else:
            print("Failed to build the TensorRT engine.")

