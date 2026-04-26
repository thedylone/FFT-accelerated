# High Performance Computing for the FFT Algorithm

The Fast Fourier Transform is a fast $O(n \log n)$ method to calculate the Discrete Fourier Transform, which is naively $O(n^2)$. Consider the scenario where we want to mulitply two polynomials of degree $\leq n-1$:

$$p(x) = a_0 + a_1 x + \dots + a_{n-1} x^{n-1}$$

$$q(x) = b_0 + b_1 x + \dots + b_{n-1} x^{n-1}$$

and we want to find $r(x) = p(x) \cdot q(x) = c_0 + c_1  x + \dots + c_{n-1} x^{2n-2}$. The naive way is to multiply everything out and collect terms with the same power of $x$. The general case gives:

$$
    r(x) = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} a_i b_j x^{i+j} = \sum_{k=0}^{2n-2} \left( \sum_{i+j=k} a_i b_j \right) x^k = \sum_{k=0}^{2n-2} \left( \sum_{i=0}^{k} a_i b_{k-i} \right) x^k
$$

The total number of terms is $\Theta(n^2)$, so this gives an algorithm that performs a quadratic number of arithmetic operations. While this seems like the problem must require quadratic time, surprisingly the FFT can achieve this in $O(n \log n)$.

FFT is used in this scenario to turn these "coefficient" polynomials into "point-value" polynomials. By the fundamental theorem of algebra, a polynomial of degree $2n - 2$ (for $r$) can be uniquely determined by $2n-1$ points. So the polynomials are computed at the same points, allowing us to do point-wise multiplication to get the "point-value" representation of $r(x)$ in $O(n)$ time. Finally, inverse FFT transforms the "point-value" $r(x)$ back to the "coefficient" $r(x)$, solving our scenario in $O(n \log n)$ time!

The key idea in FFT, specifically the Cooley-Tukey algorithm, is to **divide and conquer**. Suppose we want to evaluate a polynomial $p(x)$ with even $n$ at $x$ and $-x$. We can split $p$ into even and odd terms:

$$p(x) = {\left[a_0 + a_2 x^2 + \dots + a_{n-2} x^{n-2} \right]} + {\left[a_1 x + a_3 x^3 + \dots + a_{n-1} x^{n-1} \right]} $$

$$p(-x) = {\left[a_0 + a_2 x^2 + \dots + a_{n-2} x^{n-2} \right]} - {\left[a_1 x + a_3 x^3 + \dots + a_{n-1} x^{n-1} \right]} $$

So we can compute the even and odd parts separately, and adding them gives $p(x)$, subtracting them gives $p(-x)$. We get **two values for the price of one**!

Generalising this for $n$ points, we use the **roots of unity**: $[ \omega_{n} = e^{2\pi ij/n} ]_{j=0}^{n-1}$. To calculate ${\mathrm{even}}$ and ${\mathrm{odd}}$, this is simply another FFT on each of them, now with size $n/2$. To recap, the algorithm is given as:

```math
\begin{aligned}
& \text{Algorithm: Fast Fourier Transform} \\
& \text{function }FFT((a_0, \dots, a_{n-1}), n) & \triangleright \text{ Requires } n = 2^k \\
& \text{if } n = 1: \\
& \quad \text{return } a \\
& \omega_n \leftarrow e^{2\pi i / n} \\
& a_{\mathrm{even}} \leftarrow (a_0, a_2, \dots, a_{n-2}) \\
& a_{\mathrm{odd}} \leftarrow (a_1, a_3, \dots, a_{n-1}) \\
& y_{\mathrm{even}} \leftarrow FFT(a_{\mathrm{even}}, n/2) \\
& y_{\mathrm{odd}} \leftarrow FFT(a_{\mathrm{odd}}, n/2) \\
& \text{for } j \in \{0, 1, \dots, n/2 - 1 \}: \\
& \quad y[j] \leftarrow y_{\mathrm{even}}[j] + \omega_n^j y_{\mathrm{odd}}[j] \\
& \quad y[j+n/2] \leftarrow y_{\mathrm{even}}[j] - \omega_n^j y_{\mathrm{odd}}[j] \\
& \text{return } y
\end{aligned}
```

Pictorally, this is illustrated below:

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/DIT-FFT-butterfly.svg/960px-DIT-FFT-butterfly.svg.png" alt="FFT algorithm diagram. By Yangwenbo99 on Wikipedia" style="width: 50%; height: auto;">
  <p align="center">FFT algorithm diagram. By Yangwenbo99 on Wikipedia</p>
</p>

In practice, we implement the algorithm to be iterative (instead of recursive), and apply the modifications in-place. Bit-reversal permutation reorders the input so that the iterative butterfly stages can be applied in-place without extra bookkeeping. Butterfly stages are $\log(n)$ passes over the array. In each pass, pairs of elements are combined using a twiddle factor (a root of unity). Each level corresponds to a halved sub-problem in the recursive view.
