/// @file TorchScript Loader
/// @author Jongkuk Lim <limjk@jmarple.ai>
/// @copyright 2021 J.Marple
/// @brief TorchScript loader module.

#include "torchscript_loader/torchscript_loader.hpp"

#include <torch/torch.h>

namespace ts_loader {
ConvModule::ConvModule(const string& path, bool _is_training)
    : is_training_(_is_training) {
  if (!torch::cuda::is_available()) {
    std::cout << "GPU can not be found. Switching to CPU and float32 precision."
              << std::endl;
    device_ = torch::kCPU;
    d_type_ = torch::kFloat;
  }

  module_ = torch::jit::load(path, device_);
  module_.to(d_type_);
  module_.to(device_);

  if (!is_training_) module_.eval();
}

/// Preprocess tensor by normalization.
///
/// @param tensor: input tensor to be preprocessed
///
/// @return (tensor / 255.)
torch::Tensor ConvModule::preprocess_tensor(torch::Tensor tensor) {
  torch::Tensor out_tensor = tensor.clone();
  out_tensor.div_(255.);

  return out_tensor.toType(d_type_);
}

/// Fowrad torch tensor.
///
/// @param tensor: torch tensor (BCHW)
/// @param do_preprocess: Skp preprocess if false.
///
/// @return feature map of conv module (BCHW)
at::Tensor ConvModule::forward(const torch::Tensor& tensor,
                               bool do_preprocess) {
  std::vector<torch::jit::IValue> inputs;

  if (do_preprocess)
    inputs.push_back(preprocess_tensor(tensor.to(device_)));
  else
    inputs.push_back(tensor.toType(d_type_).to(device_));

  c10::IValue result = module_.forward(inputs);
  return result.toTuple()->elements()[0].toTensor();
}

}  // namespace ts_loader
