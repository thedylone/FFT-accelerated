import math

import jax
import jax.numpy as jnp


def bit_reverse(a: jax.Array, log2n: int) -> jax.Array:
    """Bit reversal permutation."""
    n = 1 << log2n
    indices = jnp.arange(n)
    rev = jnp.zeros(n, dtype=jnp.int32)
    for bit in range(log2n):
        rev = rev | (((indices >> bit) & 1) << (log2n - 1 - bit))
    return a[rev]


def fft(a: jax.Array, direction: int = 1) -> jax.Array:
    """Iterative Cooley-Tukey radix-2 FFT.

    direction = +1: forward FFT
    direction = -1: inverse FFT (caller must divide by n)
    """
    n = a.shape[0]
    log2n = int(math.log2(n))
    a = bit_reverse(a, log2n)

    for s in range(log2n):
        length = 1 << (s + 1)
        half = length // 2

        angle = direction * 2.0 * jnp.pi / length
        # twiddle factors for one block
        w = jnp.exp(1j * angle * jnp.arange(half))

        # reshape into blocks, apply butterflies in parallel
        a = a.reshape(-1, length)
        top = a[:, :half]
        bot = a[:, half:]
        t = bot * w
        a = jnp.concatenate([top + t, top - t], axis=1).reshape(-1)

    return a
