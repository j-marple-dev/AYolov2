"""Model converter module.

Converts PyTorch model into TorchScript, ONNX and, TensorRT.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import os
from typing import Any, Optional, Tuple, Union

import numpy as np
import onnx
import torch
from onnxsim import simplify
from torch import nn

from scripts.utils.logger import get_logger

LOGGER = get_logger(__name__)
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
        self.test_input = torch.zeros(batch_size, 3, *input_size, device=self.device)

    def dry_run(self) -> None:
        """Dry run the model for memory load purpose."""
        self.model.eval()
        self.model(self.test_input)

    @torch.no_grad()
    def to_torch_script(
        self, path: str, half: bool = False, use_cuda: bool = True
    ) -> None:
        """Export model to TorchScript.

        Args:
            path: export path.
            half: export half precision model.
            use_cuda: use cuda device.
        """
        device = torch.device(
            "cuda:0" if (half or use_cuda) else "cpu"
        )  # Half precision works on GPU only.
        test_input = self.test_input.to(device)
        self.model.to(device).eval()

        if half:
            self.model.half()
            test_input = test_input.half()

        for attr_name in dir(self.model.model[-1]):  # type: ignore
            attr = getattr(self.model.model[-1], attr_name)  # type: ignore
            if isinstance(attr, torch.Tensor):
                LOGGER.info(
                    f"Converting {attr_name} to {'half' if half else 'fp32'} precision ..."
                )
                attr.to(device)
                setattr(self.model.model[-1], attr_name, attr.half() if half else attr)  # type: ignore
            elif isinstance(attr, list) and isinstance(attr[0], torch.Tensor):
                for i in range(len(attr)):
                    LOGGER.info(
                        f"Converting {attr_name}[{i}] to {'half' if half else 'fp32'} precision ..."
                    )
                    attr[i] = attr[i].to(device)
                    attr[i] = attr[i].half() if half else attr[i]
                setattr(self.model.model[-1], attr_name, attr)  # type: ignore

        self.model(test_input)
        ts = torch.jit.trace(self.model, test_input)
        ts.save(path)

    def to_onnx(self, path: str, opset_version: int = 11, **kwargs: Any) -> None:
        """Export model to ONNX model.

        Args:
           path: export path
           opset_version: ONNX op set version.
           kwargs: keyword arguments for torch.onnx.export
        """
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
        build_nms: bool = True,
        conf_thres: float = 0.1,
        iou_thres: float = 0.6,
        top_k: int = 512,
        keep_top_k: int = 100,
    ) -> bool:
        """Convert model to TensorRT model.

        Args:
            path: export path
            opset_version: ONNX op set version. Recommended to stick with 11.
            int8: convert model to INT8 precision. int8_calibrator must be provided.
            int8_calibrator: int8 calibrator instance. (tensorrt.IInt8EntropyCalibrator2)
            build_nms: build TRT model with NMS layer.
            conf_thres: Confidence threshold for Batch NMS.
            iou_thres: IoU threshold for Batch NMS.
            top_k: top k parameter for Batch NMS.
            keep_top_k: keep top k number of bboxes for Batch NMS.

        Return:
            True if converting is success
        """
        try:
            import tensorrt as trt  # pylint: disable=import-outside-toplevel, import-error
        except ModuleNotFoundError as error:
            raise Exception("TensorRT is not installed.") from error

        trt.init_libnvinfer_plugins(None, "")

        onnx_path = f"{path}.onnx"
        self.to_onnx(onnx_path, opset_version=opset_version)

        if not os.path.isfile(onnx_path):
            return False
        with open(onnx_path, "rb") as f:
            onnx_data = f.read()

        if self.verbose > 0:
            trt_logger = trt.Logger(trt.Logger.VERBOSE)
        else:
            trt_logger = trt.Logger()

        explicit_batch = [1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]

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

            if not parser.parse(onnx_data):
                print("Failed to parse the ONNX file.")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return False

            network.get_input(0).shape = (self.batch_size,) + network.get_input(
                0
            ).shape[1:]
            # Add custom layer here if deisred.

            # builder.max_batch_size = self.TRT_MAX_BATCH_SIZE
            builder.max_batch_size = self.batch_size
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size_gib << 30
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

            profile = builder.create_optimization_profile()
            profile.set_shape(
                input="images",
                min=(
                    # self.TRT_MAX_BATCH_SIZE,
                    self.batch_size,
                    3,
                    self.test_input.shape[2],
                    self.test_input.shape[3],
                ),
                opt=(
                    # self.TRT_MAX_BATCH_SIZE,
                    self.batch_size,
                    3,
                    self.test_input.shape[2],
                    self.test_input.shape[3],
                ),
                max=(
                    # self.TRT_MAX_BATCH_SIZE,
                    self.batch_size,
                    3,
                    self.test_input.shape[2],
                    self.test_input.shape[3],
                ),
            )
            config.add_optimization_profile(profile)

            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                builder.fp16_mode = True

            if int8:
                # TODO(jeikeilim): This won't work for now.
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = int8_calibrator
                config.set_calibration_profile(profile)

            if dla_core_id >= 0:
                config.default_device_type = trt.DeviceType.DLA
                config.DLA_core = dla_core_id
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                print(f"Using DLA Core id {dla_core_id}")

            # Add Batched NMS layer  ---- START
            if build_nms:
                previous_output = network.get_output(0)
                network.unmark_output(previous_output)

                strides = trt.Dims([1, 1, 1])
                starts = trt.Dims([0, 0, 0])

                bs, num_boxes, n_out = previous_output.shape
                num_classes = n_out - 5
                shapes = trt.Dims([bs, num_boxes, 4])

                # bs, num_boxes, 4
                boxes = network.add_slice(previous_output, starts, shapes, strides)

                starts[2] = 4
                shapes[2] = 1
                # bs, num_boxes, 1
                obj_score = network.add_slice(previous_output, starts, shapes, strides)

                starts[2] = 5
                shapes[2] = num_classes
                # bs, num_boxes, num_classes
                scores = network.add_slice(previous_output, starts, shapes, strides)
                indices = network.add_constant(
                    trt.Dims([num_classes]),
                    trt.Weights(np.zeros(num_classes, np.int32)),
                )
                gather_layer = network.add_gather(
                    obj_score.get_output(0), indices.get_output(0), 2
                )
                updated_scores = network.add_elementwise(
                    gather_layer.get_output(0),
                    scores.get_output(0),
                    trt.ElementWiseOperation.PROD,
                )

                # reshape box to [bs, num_boxes, 1, 4]
                reshaped_boxes = network.add_shuffle(boxes.get_output(0))
                reshaped_boxes.reshape_dims = trt.Dims([0, 0, 1, 4])

                # add batchedNMSPlugin, inputs:[boxes:(bs, num, 1, 4), scores:(bs, num, 1)]
                trt.init_libnvinfer_plugins(trt_logger, "")
                registry = trt.get_plugin_registry()
                assert registry
                creator = registry.get_plugin_creator("BatchedNMS_TRT", "1")
                assert creator
                fc = []
                fc.append(
                    trt.PluginField(
                        "shareLocation",
                        np.array([1], dtype=int),
                        trt.PluginFieldType.INT32,
                    )
                )
                fc.append(
                    trt.PluginField(
                        "backgroundLabelId",
                        np.array([-1], dtype=int),
                        trt.PluginFieldType.INT32,
                    )
                )
                fc.append(
                    trt.PluginField(
                        "numClasses",
                        np.array([num_classes], dtype=int),
                        trt.PluginFieldType.INT32,
                    )
                )
                fc.append(
                    trt.PluginField(
                        "topK", np.array([top_k], dtype=int), trt.PluginFieldType.INT32
                    )
                )
                fc.append(
                    trt.PluginField(
                        "keepTopK",
                        np.array([keep_top_k], dtype=int),
                        trt.PluginFieldType.INT32,
                    )
                )
                fc.append(
                    trt.PluginField(
                        "scoreThreshold",
                        np.array([conf_thres], dtype=np.float32),
                        trt.PluginFieldType.FLOAT32,
                    )
                )
                fc.append(
                    trt.PluginField(
                        "iouThreshold",
                        np.array([iou_thres], dtype=np.float32),
                        trt.PluginFieldType.FLOAT32,
                    )
                )
                fc.append(
                    trt.PluginField(
                        "isNormalized",
                        np.array([0], dtype=int),
                        trt.PluginFieldType.INT32,
                    )
                )
                fc.append(
                    trt.PluginField(
                        "clipBoxes", np.array([0], dtype=int), trt.PluginFieldType.INT32
                    )
                )

                fc = trt.PluginFieldCollection(fc)
                nms_layer = creator.create_plugin("nms_layer", fc)
                layer = network.add_plugin_v2(
                    [reshaped_boxes.get_output(0), updated_scores.get_output(0)],
                    nms_layer,
                )
                layer.get_output(0).name = "num_detections"
                layer.get_output(1).name = "nmsed_boxes"
                layer.get_output(2).name = "nmsed_scores"
                layer.get_output(3).name = "nmsed_classes"
                for i in range(4):
                    network.mark_output(layer.get_output(i))
                # Add Batched NMS layer  ---- END

            engine = builder.build_engine(network, config)

        os.remove(onnx_path)
        if engine is not None:
            with open(path, "wb") as f:
                f.write(engine.serialize())
            print(f"Completed converting to TensorRT model file to {path}")
            return True
        else:
            print("Failed to build the TensorRT engine.")
            return False
