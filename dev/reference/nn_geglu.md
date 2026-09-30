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
#>  3.1340e-02 -3.3311e-03  1.7472e-01 -2.3679e-02  4.7118e-01
#> -5.1074e-02 -2.0567e-01  7.3613e-03 -1.4771e-01 -1.3894e+00
#>  1.4928e+00 -2.4993e-01  9.0271e-02 -6.3798e-02  5.5783e-02
#>  1.5537e-01  2.1246e+00  4.9160e-01  2.1933e-01  2.1598e-01
#>  9.4293e-02  5.1981e-02  8.8806e-01 -1.8190e-01 -7.9016e-03
#> -1.2226e-02 -4.1528e-02  8.6372e-02 -8.9123e-01  3.6129e-02
#>  9.2031e-01 -8.3238e-02 -8.3964e-02 -2.2392e-01  2.9647e+00
#> -1.8331e-01 -8.6243e-02  7.2827e-02 -8.0130e-02 -1.0883e+00
#>  1.2371e-01  1.2597e-01 -2.5522e-05 -4.4664e-02 -4.7554e-02
#> -4.0408e-04  5.5798e-03 -3.8037e-01 -4.1461e-02  8.4195e-02
#> [ CPUFloatType{10,5} ]
```
