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
#> -5.4389e-02 -6.3537e-02  2.3029e-01 -5.4287e-02 -8.3351e-02
#> -2.2622e-01  1.7904e-01  7.9821e-01  4.8524e-01  1.2751e-01
#>  6.3241e-03  2.0919e-02 -1.4230e-02 -3.8427e-02 -1.1956e-01
#> -2.2645e-01 -8.9345e-02  9.1197e-02  1.0389e-02 -2.6652e-02
#>  5.6980e-02  1.2952e-01  1.3438e-01 -2.6642e-01  3.1617e-02
#> -1.7767e+00 -3.2412e-03  4.2556e-02 -4.2006e-02  5.9667e-02
#>  9.4776e-02  1.5854e-02  1.1735e+00 -3.6482e-05  9.7848e-02
#>  1.0998e-01 -1.3918e-02  1.7506e-02 -4.8450e-01 -6.8079e-01
#> -7.4085e-02 -3.7317e-02  4.4020e-03 -8.4840e-02 -9.8971e-02
#> -1.0961e+00  2.1819e-02  1.1781e-02  7.7821e-03 -2.1460e-02
#> [ CPUFloatType{10,5} ]
```
