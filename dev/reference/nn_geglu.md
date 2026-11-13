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
#> -1.9712e-01 -2.1857e-01  1.0670e-01  2.5902e+00  6.3869e-02
#> -6.0518e-01 -8.7825e-05  1.0089e-01 -3.8223e+00 -3.8658e-02
#> -1.0393e-01  3.9162e-03  1.8790e-03 -2.7727e-01 -3.9321e-02
#>  1.5525e-01  3.9707e-02 -1.8010e-01 -2.7960e-03 -7.0858e-02
#> -4.5323e-03 -1.2464e+00 -7.0315e-02 -5.2036e-03  2.5810e-01
#> -5.8078e-02  1.2538e-01 -4.4297e-03  4.0175e-01 -6.4603e-02
#>  1.1634e-01  5.5783e-01 -1.9495e-01  5.0772e-02 -4.7342e-02
#>  1.2437e-01 -7.4841e-02  1.5620e-02 -1.3609e-02 -1.8968e+00
#> -3.3736e-01  4.2270e-01  2.9946e-01 -1.0481e-01 -8.4789e-01
#>  1.6523e-01 -2.3685e-01 -2.0447e+00  1.1267e-02  5.0816e-02
#> [ CPUFloatType{10,5} ]
```
