/// @file
/// @author Jongkuk Lim <limjk@jmarple.ai>
/// @copyright 2021 J.Marple
/// @brief This module demonstrates how to construct C++ project.

#include "trt_loader/trt_loader.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#include "NvCaffeParser.h"
#include "cuda/cudaMappedMemory.h"

namespace trt_loader {

int load_trt_model(const char* path, uint32_t max_batch_size,
                   const std::vector<std::string>& output_blobs) {
  std::stringstream model_stream;
  model_stream.seekg(0, model_stream.beg);

  trt_loader::Logger trt_logger;
  std::ifstream cache(path);
  if (!cache) {
    std::cout << path << " not found!" << std::endl;
    return 1;
  }

  model_stream << cache.rdbuf();

  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(trt_logger);

  if (builder != NULL) {
    bool enable_fp16 = builder->platformHasFastFp16();
    printf("platform %s FP16 support.\n",
           enable_fp16 ? "has" : "does not have");
    builder->destroy();
  }

  printf("%s loaded\n", path);

  nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(trt_logger);
  if (!infer) {
    printf("failed to create InferRuntime\n");
    return 1;
  }

  model_stream.seekg(0, std::ios::end);
  const int model_size = model_stream.tellg();
  model_stream.seekg(0, std::ios::beg);

  void* model_mem = malloc(model_size);
  if (!model_mem) {
    printf("failed to allocate %i bytes to deserialize model\n", model_size);
    return 0;
  }

  model_stream.read(reinterpret_cast<char*>(model_mem), model_size);
  nvinfer1::ICudaEngine* engine =
      infer->deserializeCudaEngine(model_mem, model_size, NULL);
  free(model_mem);

  if (!engine) {
    printf("failed to create CUDA engine\n");
    return 0;
  }

  nvinfer1::IExecutionContext* context = engine->createExecutionContext();

  if (!context) {
    printf("failed to create execution context\n");
    return 0;
  }

  printf("CUDA engine context initialized with %u bindings\n",
         engine->getNbBindings());

  const int input_index = engine->getBindingIndex("images");

  printf("%s input  binding index:  %i\n", path, input_index);

  nvinfer1::Dims input_dims = engine->getBindingDimensions(input_index);

  size_t input_size = max_batch_size * input_dims.d[0] * input_dims.d[1] *
                      input_dims.d[2] * sizeof(float);

  std::cout << path << " input dims: " << input_dims.d[0] << "x"
            << input_dims.d[1] << "x" << input_dims.d[2] << "x"
            << input_dims.d[3] << ", size=" << input_size << std::endl;

  const int num_outputs = output_blobs.size();

  std::vector<trt_loader::outputLayer> outputs;

  for (int n = 0; n < num_outputs; n++) {
    const int output_index = engine->getBindingIndex(output_blobs[n].c_str());
    printf("%s output %i %s  binding index:  %i\n", path, n,
           output_blobs[n].c_str(), output_index);
    nvinfer1::Dims output_dims = engine->getBindingDimensions(output_index);
    size_t output_size = max_batch_size * output_dims.d[0] * output_dims.d[1] *
                         output_dims.d[2] * sizeof(float);
    printf("%s output %i %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", path, n,
           output_blobs[n].c_str(), max_batch_size, output_dims.d[0],
           output_dims.d[1], output_dims.d[2], output_size);

    // allocate output memory
    void* output_cpu = NULL;
    void* output_cuda = NULL;

    if (!cudaAllocMapped(reinterpret_cast<void**>(&output_cpu),
                         reinterpret_cast<void**>(&output_cuda), output_size)) {
      printf("failed to alloc CUDA mapped memory\n");
      return 1;
    }

    trt_loader::outputLayer l;
    l.CPU = reinterpret_cast<float*>(output_cpu);
    l.CUDA = reinterpret_cast<float*>(output_cuda);
    l.size = output_size;
    l.dims.d[0] = output_dims.d[0];
    l.dims.d[1] = output_dims.d[1];
    l.dims.d[2] = output_dims.d[2];
    l.name = output_blobs[n];

    outputs.push_back(l);
  }

  printf("%s initialized.\n", path);

  return 0;
}

}  // namespace trt_loader
