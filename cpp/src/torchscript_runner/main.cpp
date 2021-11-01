/// @file TorchScript Runner
/// @author Jongkuk Lim <limjk@jmarple.ai>
/// @copyright 2021 J.Marple
/// @brief Main source code for torch script runner.

#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <string>

#include "torchscript_loader/torchscript_loader.hpp"

/// Run TorschScript model
///
/// @param argc: Number of arguments.
/// @param argv: Arguments
///
/// @return: 0
int main(int argc, char **argv) {
  for (int i = 0; i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  ts_loader::ConvModule model(argv[1]);

  std::cout << "Is CUDA available: " << torch::cuda::is_available()
            << std::endl;

  torch::Tensor test_input =
      torch::rand({32, 3, 640, 640}, model.d_type).to(model.device);

  std::cout << "Data Type: " << model.d_type << ", Device: " << model.device
            << std::endl;

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  for (int i = 0; i < 137; i++) {
    torch::Tensor x = model.forward(test_input, false);

    std::cout << "Running " << i << " step ... " << x.sizes() << std::endl;
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Time difference = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                   .count()
            << "[ms]" << std::endl;

  return 0;
}
