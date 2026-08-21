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
#> -0.1453  0.0344  0.8878 -0.1511 -0.1456
#> -0.2594  0.4595  0.1595  0.1268  0.0770
#> -0.0268 -0.0120  0.0110  0.0076  0.0149
#>  0.5697 -0.2060  0.1362 -0.0622 -0.3610
#> -0.6070 -0.0275  0.1696 -0.4019 -0.0017
#>  0.1804 -0.3124 -0.0092  1.8710 -0.0181
#> -0.1588 -0.0015 -0.9708  0.0135 -0.1280
#>  0.0535  0.0276 -0.1460  0.0450 -0.2426
#> -0.1728 -0.7905 -0.0597 -0.1604 -0.0505
#>  0.0650  0.0147 -0.0387  0.4253 -0.0914
#> [ CPUFloatType{10,5} ]
```
