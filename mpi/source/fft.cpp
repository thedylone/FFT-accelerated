#include <cmath>
#include <cassert>
#include <mpi.h>

#include "fft.h"


/// @brief in place bit reversal permutation
///
/// reorders the input so that the iterative butterfly stages can be
/// applied in place.
void bit_reverse(Vec& a) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

/// @brief iterative Cooley-Tukey radix-2 in place FFT using MPI
///
/// direction = +1: forward FFT
/// direction = -1: inverse FFT (caller must divide by n)
///
/// rank 0 owns the full vector `a`; other ranks pass an empty vector.
/// internally, data is scattered so each rank only stores n/p elements.
/// after computation, results are gathered back to rank 0.
void fft(Vec& a, int direction) {
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // broadcast size n from rank 0
    std::size_t n = a.size();
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    assert((n & (n - 1)) == 0 && "FFT size must be a power of 2");
    assert(n >= static_cast<std::size_t>(size)
           && "FFT size must be >= number of processes");

    const std::size_t local_n = n / size;
    // bit reverse, then scatter
    if (rank == 0) bit_reverse(a);
    Vec local(local_n);
    MPI_Scatter(reinterpret_cast<double*>(a.data()), static_cast<int>(local_n * 2), MPI_DOUBLE,
        reinterpret_cast<double*>(local.data()), static_cast<int>(local_n * 2), MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // local butterfly
    for (std::size_t len = 2; len <= local_n; len <<= 1) {
        const double angle = direction * 2.0 * PI / static_cast<double>(len);
        const Complex w_len(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < local_n; i += len) {
            Complex w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const Complex u = local[i + j];
                const Complex v = local[i + j + len / 2] * w;
                local[i + j] = u + v;
                local[i + j + len / 2] = u - v;
                w *= w_len;
            }
        }
    }

    // now binary exchange to compute cross-rank butterflies
    Vec recv(local_n);
    int log2_p = 0;  // log num of procs, gives number of stages
    while ((1 << log2_p) < size) ++log2_p;

    for (int d = 0; d < log2_p; ++d) {
        const std::size_t len = local_n << (d + 1);
        const int partner = rank ^ (1 << d);
        const bool is_top = ((rank >> d) & 1) == 0;  // determine the "top" rank in the pair

        MPI_Sendrecv(reinterpret_cast<double*>(local.data()), static_cast<int>(local_n * 2), MPI_DOUBLE,
            partner, 0,
            reinterpret_cast<double*>(recv.data()), static_cast<int>(local_n * 2), MPI_DOUBLE,
            partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // position of local[0] within its butterfly block's half
        const int block_rank = rank & ((1 << (d + 1)) - 1);
        const std::size_t twiddle_base =
            is_top ? static_cast<std::size_t>(block_rank) * local_n
                : static_cast<std::size_t>(block_rank - (1 << d)) * local_n;

        const double base_angle = direction * 2.0 * PI / static_cast<double>(len);
        const Complex w_step(std::cos(base_angle), std::sin(base_angle));
        const double theta0 = base_angle * static_cast<double>(twiddle_base);
        Complex w(std::cos(theta0), std::sin(theta0));

        for (std::size_t j = 0; j < local_n; ++j) {
            if (is_top) {
                const Complex u = local[j];
                const Complex v = recv[j] * w;
                local[j] = u + v;
            } else {
                const Complex u = recv[j];
                const Complex v = local[j] * w;
                local[j] = u - v;
            }
            w *= w_step;
        }
    }

    // gather back to rank 0
    MPI_Gather(reinterpret_cast<double*>(local.data()), static_cast<int>(local_n * 2), MPI_DOUBLE,
            reinterpret_cast<double*>(a.data()), static_cast<int>(local_n * 2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
