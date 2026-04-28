#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#include "fft.h"


/// @brief correct if FFT then IFFT recovers the original signal
bool correctness_check(std::size_t n, int block_size) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Vec original(n);
    for (auto& x : original) x = {dist(rng), dist(rng)};

    Vec signal = original;
    fft(signal, +1);
    fft(signal, -1);
    for (auto& x : signal) x /= static_cast<double>(n);

    constexpr double tol = 1e-9;
    for (std::size_t i = 0; i < n; ++i)
        if (std::abs(signal[i] - original[i]) > tol) return false;
    return true;
}

int main(int argc, char* argv[]) {
    int block_size = 512;
    if (argc > 1) block_size = std::atoi(argv[1]);
    if (block_size <= 0 || block_size > 1024 || block_size % 32 != 0) {
        std::cerr << "Error: block_size must be a multiple of 32 in [32, 1024]\n";
        return 1;
    }
    std::cout << "Using block size: " << block_size << "\n\n";

    // Print GPU info
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::cout << "GPU: " << prop.name
              << "  SMs: " << prop.multiProcessorCount
              << "  Mem: " << prop.totalGlobalMem / (1 << 20) << " MiB\n\n";

    // check for correctness
    for (std::size_t n : {1u<<10, 1u<<16, 1u<<20}) {
        bool ok = correctness_check(n, block_size);
        std::cout << "Correctness check (n=" << n << "): "
                  << (ok ? "PASSED" : "FAILED") << "\n";
    }
    std::cout << "\n";

    // benchmark performance
    std::cout << "block_size,n,time_ms\n";

    for (int exp = 10; exp <= 26; ++exp) {
        const std::size_t n = 1u << exp;

        std::mt19937_64 rng(123);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Vec a(n);
        for (auto& x : a) x = {dist(rng), 0.0};

        // warm-up 
        Vec tmp = a; 
        auto t0 = std::chrono::high_resolution_clock::now();
        fft(tmp, +1, block_size);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        const int REPS = std::max(1000 / ms, 3.0);
        // time just the copy cost
        t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPS; ++r) {
            Vec temp = a;
            (void)temp;
        }
        t1 = std::chrono::high_resolution_clock::now();
        double copy_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
                    / REPS;
        // time copy + fft
        t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPS; ++r) {
            Vec run = a; fft(run, +1, block_size);
        }
        t1 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
                    / REPS;
        std::cout << block_size << "," << n << "," << ms - copy_ms << "\n";
        std::cout.flush();
    }
    return 0;
}