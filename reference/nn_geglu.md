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
#>  0.0702  0.5824 -0.0148 -0.3396  0.1149
#> -0.0229 -0.0354 -0.1069  0.0008 -0.1726
#> -0.1261  0.0222 -0.1275 -0.0449 -0.2198
#> -0.3329  0.1446 -0.0321 -4.5070 -0.1416
#> -0.2619  0.0470  1.5812 -0.0567 -0.5928
#>  0.2794 -0.5343 -0.0235 -0.0694 -0.0241
#>  0.0008  0.1562 -0.1724 -0.7705  0.1023
#>  0.1311 -0.7396  0.0018  0.1862  0.0026
#> -0.2623 -0.0661 -0.0113 -0.2421  0.0178
#> -0.0232  2.7751  1.9706 -0.1020 -0.0668
#> [ CPUFloatType{10,5} ]
```
