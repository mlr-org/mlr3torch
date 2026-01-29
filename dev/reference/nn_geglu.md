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
#> -2.1988e-01  2.3768e-01  6.2995e-03  3.9827e-01 -2.5512e-01
#> -9.8992e-03  6.8729e-03 -3.6842e-02  1.4539e-01 -4.7830e-02
#>  4.4145e-02 -3.6310e-01  6.1719e-02 -1.3550e-02  7.7087e-03
#>  2.6183e-01 -3.7488e-02 -2.0366e-02 -8.6995e-02  6.8289e-03
#> -1.2990e-01  5.1119e-01  1.3022e+00 -1.1559e+00 -1.0124e+00
#> -5.8668e-02 -2.3469e-01  5.6778e-03 -1.2681e+00 -9.4738e-02
#> -9.3919e-02 -7.2981e-02 -5.9784e-02  2.3321e-02 -3.6557e-02
#> -9.0041e-02 -6.2381e-01 -1.7216e-01 -2.0146e+00 -1.9885e+00
#> -8.6139e-02 -2.0301e-01 -2.9420e-01 -7.2027e-02 -4.2880e-02
#> -1.7116e-05 -1.8886e-01 -3.4989e-01  6.6770e-02 -8.4809e-02
#> [ CPUFloatType{10,5} ]
```
