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
#>  0.4499 -1.0198  0.1729  0.0310 -0.4190
#>  0.2118  0.0438  0.5492  0.4709 -0.0436
#>  0.3741 -0.0599 -0.3455  0.3625 -0.0992
#>  0.2810  2.2715 -0.1532  0.1340 -0.0570
#>  0.0681  0.0625  0.0121 -0.0051  0.1046
#>  0.2319  0.0227  0.0059  0.5772  0.1399
#> -0.0473 -0.0422 -0.3499 -0.4412 -0.2604
#>  0.1566  0.0318  0.0172 -2.8097 -0.2595
#>  0.0402 -0.1147 -0.1820 -0.1179 -0.0482
#> -2.3121  0.0461 -0.1367  0.2302 -2.1281
#> [ CPUFloatType{10,5} ]
```
