#include <iostream>
#include <random>
#include <chrono>
#include <mpi.h>

#include "fft.h"


/// @brief correct if FFT then IFFT recovers the original signal
bool correctness_check(std::size_t n) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Vec signal;
    Vec original;
    if (rank == 0) {
        signal.resize(n);
        original.resize(n);
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        for (auto& x : original) x = {dist(rng), dist(rng)};
        signal = original;
    }
    fft(signal, +1);
    fft(signal, -1);

    bool ok = true;
    if (rank == 0) {
        for (auto& x : signal) x /= static_cast<double>(n);

        constexpr double tol = 1e-9;
        for (std::size_t i = 0; i < n; ++i)
            if (std::abs(signal[i] - original[i]) > tol) return false;
    }
    MPI_Bcast(&ok, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return ok;
}

int main(int argc, char *argv[]) {
    // Optionally override thread count from the command line
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cout << "Using " << size << " MPI processes\n\n";

    // check for correctness
    for (std::size_t n : {1u<<10, 1u<<16, 1u<<20}) {
        bool ok = correctness_check(n);
        if (rank == 0)
            std::cout << "Correctness check (n=" << n << "): "
                    << (ok ? "PASSED" : "FAILED") << "\n";
    }
    if (rank == 0) std::cout << "\n";

    // benchmark performance
    if (rank == 0) std::cout << "processes,n,time_ms\n";
    Vec a;

    for (int exp = 10; exp <= 26; ++exp) {
        const std::size_t n = 1u << exp;
        if (rank == 0) {
            std::mt19937_64 rng(123);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            a.resize(n);
            for (auto& x : a) x = {dist(rng), 0.0};
        }

        // warm-up
        Vec tmp = a;
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = MPI_Wtime();
        fft(tmp, +1);
        MPI_Barrier(MPI_COMM_WORLD);
        double ms = (MPI_Wtime() - t0) * 1e3;

        int REPS = std::max(1000 / ms, 3.0);
        MPI_Bcast(&REPS, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // time just the copy cost
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int r = 0; r < REPS; ++r) {
            Vec temp = a;
            (void)temp;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double copy_ms = (MPI_Wtime() - t0) * 1e3 / REPS;
        // time copy + fft
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int r = 0; r < REPS; ++r) {
            Vec run = a; fft(run, +1);
        }
        ms = (MPI_Wtime() - t0) * 1e3 / REPS;
        if (rank == 0)
            std::cout << size << "," << n << "," << ms - copy_ms << "\n";
    }
    MPI_Finalize();
    return 0;
}