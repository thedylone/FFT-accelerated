import sys
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
    backend = sys.argv[1] if len(sys.argv) > 1 else jax.default_backend()
    print(f"Using JAX backend: {backend}\n")

    # check for correctness
    for n in [1 << 10, 1 << 16, 1 << 20]:
        ok = correctness_check(n)
        print(f"Correctness check (n={n}): {'PASSED' if ok else 'FAILED'}")
    print()

    device = jax.devices(backend)[0]
    is_gpu = backend != "cpu"

    # jit compile the fft to the chosen device
    fft_jit = jax.jit(fft, static_argnames=("direction",), device=device)

    # benchmark performance
    print(f"backend,n,time_ms")

    import numpy as np_cpu

    for exp in range(10, 27):
        n = 1 << exp

        rng = np_cpu.random.default_rng(123)
        a_host = rng.standard_normal(n) + 0j  # host numpy array

        # warm-up (triggers JIT compilation)
        a_dev = jax.device_put(a_host, device)
        tmp = fft_jit(a_dev, direction=1)
        tmp.block_until_ready()
        if is_gpu:
            np_cpu.array(tmp)

        t0 = time.perf_counter()
        a_dev = jax.device_put(a_host, device)
        tmp = fft_jit(a_dev, direction=1)
        tmp.block_until_ready()
        if is_gpu:
            np_cpu.array(tmp)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1e3

        reps = max(int(1000 / ms), 3)

        # time just the host copy cost (equivalent to Vec run = a in C++)
        t0 = time.perf_counter()
        for _ in range(reps):
            a_copy = a_host.copy()
        t1 = time.perf_counter()
        copy_ms = (t1 - t0) * 1e3 / reps

        # time copy + transfer + fft (+ D2H for GPU)
        t0 = time.perf_counter()
        for _ in range(reps):
            a_copy = a_host.copy()
            a_dev = jax.device_put(a_copy, device)
            result = fft_jit(a_dev, direction=1)
            result.block_until_ready()
            if is_gpu:
                np_cpu.array(result)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1e3 / reps

        print(f"{backend},{n},{ms - copy_ms:.4f}")


if __name__ == "__main__":
    main()
