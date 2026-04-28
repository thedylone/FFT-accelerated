#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <iostream>

#include "fft.h"


#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/// @brief Utility function to compute the ceiling of integer division
/// @tparam T integer type
/// @param a numerator
/// @param b denominator
/// @return ceiling of a / b
template <typename T>
__host__ __device__ __forceinline__ T cdiv(T a, T b) {
  return (a + b - 1) / b;
}

/// @brief in place bit reversal permutation
///
/// each thread computes its own rev[i], no dependencies
/// swaps only when i < rev[i]
__global__ void bit_reverse_kernel(cuDoubleComplex* __restrict__ a, int n, int log2n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // bit reverse i
    unsigned r = 0u, x = (unsigned)i;
    for (int b = 0; b < log2n; ++b) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
    }
    if ((int)r > i) {  // swap
        cuDoubleComplex tmp = a[i];
        a[i] = a[r];
        a[r] = tmp;
    }
}

/// @brief one butterfly stage for a given sub-transform length `len`
///
/// each thread handles one butterfly pair, there are always n/2 pairs
__global__ void butterfly_kernel(cuDoubleComplex* __restrict__ a, int n, int half, double angle) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    int group = tid / half;
    int j = tid % half;
    int base = group * (half << 1);   // = group * len
    
    // compute twiddle w = e^{i * angle * j} = cos(angle*j) + i*sin(angle*j)
    double s, c;
    sincos(angle * j, &s, &c);
    cuDoubleComplex w = make_cuDoubleComplex(c, s);

    cuDoubleComplex u = a[base + j];
    cuDoubleComplex v = cuCmul(a[base + j + half], w);

    a[base + j] = cuCadd(u, v);
    a[base + j + half] = cuCsub(u, v);
}

/// @brief iterative Cooley-Tukey radix-2 in place FFT on the GPU
///
/// direction = +1: forward FFT
/// direction = -1: inverse FFT (caller must divide by n)
void fft(Vec& a, int direction, int block_size) {
    assert(block_size >= 32 && block_size <= 1024 && "block_size must be between 32 and 1024");
    assert(block_size % 32 == 0 && "block_size must be a multiple of 32");
    const int n = a.size();
    assert((n & (n - 1)) == 0 && "FFT size must be a power of 2");

    int log2n = 0;
    while ((1u << log2n) < n) ++log2n;

    // allocate and copy to device
    cuDoubleComplex* d_a;
    CUDA_CHECK(cudaMalloc(&d_a, (size_t)n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), (size_t)n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    bit_reverse_kernel<<<cdiv(n, block_size), block_size>>>(d_a, n, log2n);
    CUDA_CHECK(cudaGetLastError());

    for (int len = 2; len <= n; len <<= 1) {
        int half  = len >> 1;
        double angle = direction * 2.0 * PI / (double)len;
        butterfly_kernel<<<cdiv(n / 2, block_size), block_size>>>(d_a, n, half, angle);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(a.data(), d_a,
                          (size_t)n * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
}
