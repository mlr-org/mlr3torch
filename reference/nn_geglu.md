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
#> -0.7783  0.1117 -2.8767  0.5250 -0.0673
#>  0.2147  0.0591  0.1692  0.1411 -0.0373
#> -0.0312  0.0014  0.3201  0.6992  0.1724
#> -0.0368 -0.0435 -0.0610 -0.0435 -0.1876
#>  0.0689 -0.1162 -0.0379  0.0013  0.1429
#>  0.1152 -0.0233 -1.1344 -0.0105 -0.2263
#>  0.1424  0.0133  0.0801  0.1565 -0.2252
#>  0.0101  0.3841 -1.8572 -0.0356  0.5556
#>  0.0376 -0.5183 -0.2071  0.2026 -0.0177
#> -0.0496 -0.4650 -0.2010  0.0015  0.2909
#> [ CPUFloatType{10,5} ]
```
