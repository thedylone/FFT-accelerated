cmake_minimum_required(VERSION 3.16)

project(fft_mpi LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math -fopenmp")

find_package(MPI REQUIRED)

add_library(fft
    source/fft.cpp)
target_include_directories(fft PUBLIC include)
target_link_libraries(fft PUBLIC MPI::MPI_CXX)

add_executable(main
    source/main.cpp
)
target_link_libraries(main PUBLIC fft)
