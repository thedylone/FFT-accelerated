import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fft import fft


def correctness_check(n: int) -> bool:
    """Correct if FFT then IFFT recovers the original signal."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    original = jax.random.normal(k1, (n,)) + 1j * jax.random.normal(k2, (n,))

    signal = fft(original, +1)
    signal = fft(signal, -1) / n

    return float(jnp.max(jnp.abs(signal - original))) < 1e-6


def main():
    backend = jax.default_backend()
    print(f"Using JAX backend: {backend}\n")

    # check for correctness
    for n in [1 << 10, 1 << 16, 1 << 20]:
        ok = correctness_check(n)
        print(f"Correctness check (n={n}): {'PASSED' if ok else 'FAILED'}")
    print()

    # jit compile the fft
    fft_jit = jax.jit(fft, static_argnames=("direction",))

    # benchmark performance
    print("n,time_ms")

    for exp in range(10, 27):
        n = 1 << exp

        key = jax.random.PRNGKey(123)
        a = jax.random.normal(key, (n,), dtype=jnp.float64).astype(
            jnp.complex128
        )

        # warm-up (triggers JIT compilation)
        tmp = fft_jit(a, direction=1)
        tmp.block_until_ready()

        t0 = time.perf_counter()
        tmp = fft_jit(a, direction=1)
        tmp.block_until_ready()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1e3

        reps = max(int(1000 / ms), 3)

        t0 = time.perf_counter()
        for _ in range(reps):
            result = fft_jit(a, direction=1)
            result.block_until_ready()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1e3 / reps

        print(f"{n},{ms:.4f}")


if __name__ == "__main__":
    main()
