#include <cmath>
#include <cassert>

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

/// @brief iterative Cooley-Tukey radix-2 in place FFT
///
/// direction = +1: forward FFT
/// direction = -1: inverse FFT (caller must divide by n)
void fft(Vec& a, int direction) {
    const std::size_t n = a.size();
    assert((n & (n - 1)) == 0 && "FFT size must be a power of 2");

    bit_reverse(a);

    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double angle = direction * 2.0 * PI / static_cast<double>(len);
        const Complex w_len(std::cos(angle), std::sin(angle));

        for (std::size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const Complex u = a[i + j];
                const Complex v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= w_len;
            }
        }
    }
}
