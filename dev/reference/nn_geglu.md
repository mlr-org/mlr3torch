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
#> -0.1544 -0.9118 -0.1697 -0.0402 -0.0839
#>  0.0518 -0.8415  1.0535 -0.0205  0.1252
#>  0.0251 -0.0773  0.0694 -1.7850  0.0789
#> -1.4014 -0.0048 -0.0032 -0.0695  0.0227
#> -0.2125  0.1165 -0.2073  0.5414 -0.1794
#> -0.0711 -2.4124 -0.2332 -0.1458  0.1193
#> -0.2921 -0.1531  0.1622 -0.1030 -3.5904
#> -0.0986  0.1373 -0.0052  0.0053 -0.0387
#>  0.0973 -0.1226 -0.0741 -0.0183 -0.0836
#> -0.3038  0.1122  0.2138  0.0492  0.0869
#> [ CPUFloatType{10,5} ]
```
