cmake_minimum_required(VERSION 3.16)

project(fft_omp LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math")

add_library(fft
    source/fft.cpp)
target_include_directories(fft PUBLIC include)

add_executable(main
    source/main.cpp
)
target_link_libraries(main PUBLIC fft)
