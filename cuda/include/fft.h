#pragma once

#include <vector>
#include <complex>

using Complex = std::complex<double>;
using Vec = std::vector<Complex>;

// static constexpr double PI = std::acos(-1.0);
static constexpr double PI = 3.14159265358979323846;

/// @brief iterative Cooley-Tukey radix-2 in place FFT
///
/// direction = +1: forward FFT
/// direction = -1: inverse FFT (caller must divide by n)
void fft(Vec& a, int direction = 1);
