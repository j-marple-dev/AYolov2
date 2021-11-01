/// @file
/// @author Jongkuk Lim <limjk@jmarple.ai>
/// @copyright 2021 J.Marple
/// @brief This module loads TorchScript model.

#ifndef CPP_INCLUDE_TORCHSCRIPT_LOADER_TORCHSCRIPT_LOADER_HPP_
#define CPP_INCLUDE_TORCHSCRIPT_LOADER_TORCHSCRIPT_LOADER_HPP_

#include <torch/nn.h>
#include <torch/script.h>

#include <string>
#include <vector>

// #include <opencv2/core.hpp>

using std::string;
using torch::jit::script::Module;

/// Torch Script Loader module.
namespace ts_loader {
/// LibTorch conv module with JIT script.
class ConvModule {
 public:
  explicit ConvModule(const string& path, bool _is_training = false);

  // Convert OpenCV Mat(HWC) to torch Tensor (1CHW)
  // mat is assumed to be BGR and 3 channels.
  // torch::Tensor convert_input_from_cv(const cv::Mat& mat);
  torch::Tensor preprocess_tensor(torch::Tensor tensor);
  // at::Tensor forward(const cv::Mat& img);          // NOLINT
  at::Tensor forward(const torch::Tensor& tensor,  // NOLINT
                     bool do_preprocess = true);

  /// Loaded module from JIT script file. (Readonly)
  Module& module = module_;
  /// Torch device to run computation. (Readonly)
  const torch::Device& device = device_;
  /// Torch data type to compute. (Readonly)
  const torch::Dtype& d_type = d_type_;

 private:
  /// Loaded module from JIT script file.
  Module module_;

  /// Torch device to run computation.
  // torch::Dtype d_type_ = torch::kHalf;
  torch::Dtype d_type_ = torch::kFloat;
  /// Torch data type to compute.
  torch::Device device_ = torch::kCUDA;
  bool is_training_ = false;
};
}  // namespace ts_loader

#endif  // CPP_INCLUDE_TORCHSCRIPT_LOADER_TORCHSCRIPT_LOADER_HPP_
