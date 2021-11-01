cmake_minimum_required(VERSION 3.20.5)
project(
	ayolov2_trt
	VERSION 0.1.0
)

option(test "Build all tests." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS} -fPIC")

# CUDA
find_package(CUDA REQUIRED)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# TensorRT
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# LibTorch
set(Torch_DIR $ENV{HOME}/libs/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

include_directories( include )
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library( ${PROJECT_NAME}_lib ${SOURCES} )

target_link_libraries( ${PROJECT_NAME}_lib ${TORCH_LIBRARIES} )

add_executable( ${PROJECT_NAME}
    # src/tensorrt_runner/main.cpp
    src/torchscript_runner/main.cpp
)

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)

target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} nvcaffe_parser)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if (test)
    add_subdirectory("/usr/src/gtest" ${CMAKE_BINARY_DIR}/gtest)
    include_directories(${GTEST_INCLUDE_DIRS})
	enable_testing()

    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

    add_executable(test-second_module tests/test_second_module.cpp)

    target_link_libraries(test-second_module gtest gtest_main)
    target_link_libraries(test-second_module ${PROJECT_NAME}_lib)

    add_test(NAME SecondModuleTest COMMAND test-second_module)
endif()
