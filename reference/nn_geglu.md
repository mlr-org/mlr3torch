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
#> -4.8459  6.6107  0.0414 -0.2062  0.0625
#>  0.1638 -0.1922 -0.0189 -0.0539 -0.2059
#>  0.0650  0.4203  0.0517 -0.0331 -0.0636
#> -0.0226 -0.7494 -1.4112  0.1820  0.0413
#> -0.4667  0.1827  0.0579  0.1586 -0.0503
#> -0.1466 -0.0169  0.2703  0.3384  0.2330
#>  1.9109  0.1707  0.0086  0.1778  0.0630
#> -0.0032 -1.1123 -0.0302 -2.2600  0.0350
#> -0.1446 -0.0477  1.2090 -0.1811  0.0146
#> -0.1613 -0.1386 -0.1434 -0.0331 -0.1574
#> [ CPUFloatType{10,5} ]
```
