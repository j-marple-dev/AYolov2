/// @file
/// @author Jongkuk Lim <limjk@jmarple.ai>
/// @copyright 2021 J.Marple
/// @brief This module loads TensorRT model.

#ifndef CPP_INCLUDE_TRT_LOADER_TRT_LOADER_HPP_
#define CPP_INCLUDE_TRT_LOADER_TRT_LOADER_HPP_

#include <stdio.h>

#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"

/// First module description here
namespace trt_loader {
/**
 * Logger class for GIE info/warning/errors
 */
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) override {
    if (severity != Severity::kINFO /*|| mEnableDebug*/)
      printf("[AYolov2] %s\n", msg);
  }
};

int load_trt_model(const char* path, uint32_t max_batch_size,
                   const std::vector<std::string>& output_blobs);

struct outputLayer {
  std::string name;
  nvinfer1::Dims3 dims;
  uint32_t size;
  float* CPU;
  float* CUDA;
};
/// Print 'Hello' n_repeat times.
///
/// @param n_repeat: Number of repeatition.
void demo_print_hello(int n_repeat);
}  // namespace trt_loader

#endif  // CPP_INCLUDE_TRT_LOADER_TRT_LOADER_HPP_
