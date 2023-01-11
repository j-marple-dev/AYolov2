"""TensorRT Runner for YOLOv5.

- Author: Haneol Kim, Jongkuk Lim
- Contact: hekim@jmarple.ai, limjk@jmarple.ai
"""
import atexit
import time
from typing import List, Optional, Tuple, Union

import nvidia.dali
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt
import torch


def test_pycuda_install() -> None:
    """Test pycuda is installed."""
    cuda.init()
    print("CUDA device query (PyCUDA version) \n")
    print("Detected {} CUDA Capable device(s) \n".format(cuda.Device.count()))
    for i in range(cuda.Device.count()):

        gpu_device = cuda.Device(i)
        print("Device {}: {}".format(i, gpu_device.name()))
        compute_capability = float("%d.%d" % gpu_device.compute_capability())
        print("\t Compute Capability: {}".format(compute_capability))
        print(
            "\t Total Memory: {} megabytes".format(
                gpu_device.total_memory() // (1024**2)
            )
        )

        # The following will give us all remaining device attributes as seen
        # in the original deviceQuery.
        # We set up a dictionary as such so that we can easily index
        # the values using a string descriptor.

        device_attributes_tuples = gpu_device.get_attributes().items()
        device_attributes = {}

        for k, v in device_attributes_tuples:
            device_attributes[str(k)] = v

        num_mp = device_attributes["MULTIPROCESSOR_COUNT"]

        # Cores per multiprocessor is not reported by the GPU!
        # We must use a lookup table based on compute capability.
        # See the following:
        # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

        cuda_cores_per_mp = {
            5.0: 128,
            5.1: 128,
            5.2: 128,
            6.0: 64,
            6.1: 128,
            6.2: 128,
            7.5: 128,
        }[compute_capability]

        print(
            "\t ({}) Multiprocessors, ({}) CUDA Cores / Multiprocessor: {} CUDA Cores".format(
                num_mp, cuda_cores_per_mp, num_mp * cuda_cores_per_mp
            )
        )

        device_attributes.pop("MULTIPROCESSOR_COUNT")

        for k in device_attributes.keys():
            print("\t {}: {}".format(k, device_attributes[k]))


def torch_dtype_to_trt(dtype: torch.dtype) -> trt.DataType:
    """Convert torch dtype to TensorRT DataType."""
    if dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError("%s is not supported by tensorrt" % dtype)


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    """Convert TensorRT DataType to torch dtype."""
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_to_trt(
    device: torch.device,
) -> Union[trt.TensorLocation, TypeError]:
    """Convert torch device to trt device."""
    if device.type == torch.device("cuda").type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device("cpu").type:
        return trt.TensorLocation.HOST
    else:
        return TypeError("%s is not supported by tensorrt" % device)


def torch_device_from_trt(device: trt.TensorLocation) -> Union[torch.device, TypeError]:
    """Convert trt device to torch device."""
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


class TrtWrapper(object):
    """TensorRT wrapper class."""

    def __init__(
        self,
        engine: Union[trt.ICudaEngine, str],
        batch_size: int,
        device: torch.device,
        torch_input: bool = True,
    ) -> None:
        """Output assumed to be torch Tensor(GPU).

        Input assumed to be dali_tensor(GPU). if torch_input is set, Input is assumed to be torch Tensor(GPU),

        Args:
            engine: TensorRT model path or deserialized engine.
            batch_size: batch size to use.
            device: torch device to run TensorRT model
            torch_input: declare that input of __call__ will be from
                torch.Tensor, otherwise it is assumed to be dali_tensor.
        """
        self.batch_size = batch_size
        self.torch_device = device  # torch compatability
        self.trt_device = torch_device_to_trt(self.torch_device)

        # Create a Context on this device,
        # Do we need cfx?
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()

        # Deserialize the engine from file
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        if isinstance(engine, str):
            # Load plugin
            trt.init_libnvinfer_plugins(None, "")

            with open(engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            self.engine = engine

        self.context = self.engine.create_execution_context()

        self.input_names = self._trt_input_names()
        self.output_names = self._trt_output_names()
        print("[Engine Info]")

        print("Input")
        for name in self.input_names:
            print(f"{name}: {self.engine.get_binding_shape(name)}")

        print("Output")
        for name in self.output_names:
            print(f"{name}: {self.engine.get_binding_shape(name)}")

        self.bindings: List[int] = []
        self._create_input_buffers()  # Slow point
        self._create_output_buffers()  # Slow point

        torch.cuda.init()
        torch.cuda.synchronize(self.torch_device)
        torch.cuda.stream(self.stream)

        # dstroy at exit
        atexit.register(self.destroy)

    def get_stream(self) -> cuda.Stream:
        """Get CUDA stream."""
        return self.stream

    def _create_input_buffers(self) -> None:
        """Create input buffers."""
        t0 = time.time()
        self.inputs_ptr: List[Optional[int]] = [None] * len(self.input_names)
        for i, name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(name)
            shape = self.engine.get_binding_shape(idx)
            trt_type = self.engine.get_binding_dtype(idx)
            size = trt.volume(shape) * self.engine.max_batch_size
            np_type = trt.nptype(trt_type)

            # dummy host
            host_mem = cuda.pagelocked_empty(size, np_type)
            print(f"1:6: {time.time()-t0}")  # Slow point
            # alloc gpu mem
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.inputs_ptr[i] = int(device_mem)
            self.bindings.append(int(device_mem))

    def _create_output_buffers(self) -> None:
        """Create output buffers."""
        t0 = time.time()
        self.outputs_tensor = [torch.empty(1) for _ in range(len(self.output_names))]
        self.outputs_ptr = [t.data_ptr() for t in self.outputs_tensor]

        for i, name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(name)
            shape = self.engine.get_binding_shape(idx)
            trt_type = self.engine.get_binding_dtype(idx)

            # size = trt.volume(shape) * self.engine.max_batch_size
            torch_type = torch_dtype_from_trt(trt_type)

            empty_ = torch.empty(
                size=tuple(shape), dtype=torch_type, device=self.torch_device
            )
            print(f"2:7: {time.time()-t0}")  # Slow point
            self.outputs_tensor[i] = empty_
            self.outputs_ptr[i] = empty_.data_ptr()
            self.bindings.append(empty_.data_ptr())

    def _input_binding_indices(self) -> list:
        """Bind input indices."""
        return [
            i
            for i in range(self.engine.num_bindings)
            if self.engine.binding_is_input(i)
        ]

    def _output_binding_indices(self) -> list:
        """Bind output indices."""
        return [
            i
            for i in range(self.engine.num_bindings)
            if not self.engine.binding_is_input(i)
        ]

    def _trt_input_names(self) -> list:
        """Get trt input names."""
        return [self.engine.get_binding_name(i) for i in self._input_binding_indices()]

    def _trt_output_names(self) -> list:
        """Get trt output names."""
        return [self.engine.get_binding_name(i) for i in self._output_binding_indices()]

    def __call__(
        self,
        imgs: Union[torch.Tensor, nvidia.dali.backend_impl.TensorListGPU],
        raw_out: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward TensorRT model.

        Args:
            imgs: image from torch.Tensor or dali_tensor
            raw_out: return raw outputs.

        Return:
            ((batch_size, n_detect, 6) x1y1x2y2, object_score, class_id,
              (batch_size,) n_detection) or
            List[torch.Tensor] if raw_out is set
        """
        # Data transfer
        self.cfx.push()

        # reset output
        for output_tensor in self.outputs_tensor:
            if output_tensor is not None:
                output_tensor.fill_(0.0)

        # cpy bindings
        bindings = self.bindings

        # assumes single inputs
        # change input bindings to new torch tensor
        idx = self.engine.get_binding_index(self.input_names[0])
        bindings[idx] = int(imgs.data_ptr())
        """
        else:
            # DALI copy to cuda
            imgs.copy_to_external(ptr=self.inputs_ptr[0], cuda_stream=self.stream)
        """
        # Run inferecne
        # Difference between v2?(batch_size)
        # self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.context.execute_async(
            batch_size=self.batch_size,
            bindings=bindings,
            stream_handle=self.stream.handle,
        )

        # Synchronize the stream
        self.stream.synchronize()
        self.cfx.pop()

        if raw_out:
            return self.outputs_tensor
        else:
            return (
                torch.cat(
                    (
                        self.outputs_tensor[1],
                        self.outputs_tensor[2].unsqueeze(-1),
                        self.outputs_tensor[3].unsqueeze(-1),
                    ),
                    -1,
                ),
                self.outputs_tensor[0],
            )

    def destroy(self) -> None:
        """Remove any context from the top of the contex stack, deactivating it."""
        self.cfx.pop()

    def __del__(self) -> None:
        """Free CUDA memories."""
        del self.context
        del self.engine
        del self.stream
        del self.cfx
        del self.bindings
        del self.outputs_tensor
