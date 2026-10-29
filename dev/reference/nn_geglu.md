# GeGLU Module

This module implements the Gaussian Error Linear Unit Gated Linear Unit
(GeGLU) activation function. It computes \\\text{GeGLU}(x, g) = x \cdot
\text{GELU}(g)\\ where \\x\\ and \\g\\ are created by splitting the
input tensor in half along the last dimension.

## Usage

``` r
nn_geglu()
```

## References

Shazeer N (2020). “GLU Variants Improve Transformer.” 2002.05202,
<https://arxiv.org/abs/2002.05202>.

## Examples

``` r
x = torch::torch_randn(10, 10)
glu = nn_geglu()
glu(x)
#> torch_tensor
#> -2.2021e-01 -4.0791e-02 -6.4062e-02 -3.7745e-02 -7.2095e-02
#> -1.8120e-02 -6.8507e-02 -1.7414e-01  1.9195e-01 -4.2187e-02
#>  2.7501e-01 -3.3227e-02  7.5312e-01 -4.4877e-03  5.1889e-02
#> -5.3991e-01  3.2192e-01 -2.1979e-02 -5.6565e-01 -2.0436e-02
#>  1.2637e-05 -8.2468e-05 -2.2977e-02 -6.7194e-02 -2.0217e-01
#> -8.1859e-01  4.4166e-01  4.5619e-01 -6.4049e-02  1.9825e-01
#> -1.6321e+00  2.7398e-02  2.1235e-01 -1.0270e+00  9.8887e-01
#> -3.6491e-02  7.1832e-03 -3.3670e-05  1.0470e-01  5.1988e-02
#>  9.3047e-03  8.1857e-02  3.0888e-02  1.1351e+00  1.6836e-01
#> -1.4935e-01  8.2055e-02 -1.1404e-01  4.0933e-01 -1.4304e+00
#> [ CPUFloatType{10,5} ]
```
