cmake_minimum_required(VERSION 3.18)

project(fft_cuda LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

add_executable(main
    source/fft.cu
    source/main.cu
)
target_include_directories(main PUBLIC include)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)