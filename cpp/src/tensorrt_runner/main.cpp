/// @file TensorRT Runner
/// @author Jongkuk Lim <limjk@jmarple.ai>
/// @copyright 2021 J.Marple
/// @brief Main source code.

#include <iostream>
#include <string>

#include "trt_loader/trt_loader.hpp"

/// Run template program
///
/// @param argc: Number of arguments.
/// @param argv: Arguments
///
/// @return: 0
int main(int argc, char **argv) {
  for (int i = 0; i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }

  trt_loader::load_trt_model(argv[1], 1, {"output"});

  return 0;
}
