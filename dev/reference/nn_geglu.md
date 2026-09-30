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
#>  0.6797 -0.1630 -0.0490 -0.0331  0.1516
#>  0.4249 -0.0189 -0.2207  0.0098 -0.1575
#>  0.1751  0.0032  0.0517  0.1077  1.0697
#>  0.1382 -0.0891  0.0212 -0.0116 -0.0025
#> -0.0854  0.6920  1.6577 -0.1282 -0.1076
#>  0.0043  0.1060 -0.3317 -1.3818  0.7146
#> -0.3655 -0.2135 -0.0252  0.0025  0.6497
#>  0.1809  0.3025  0.1977 -0.0905  0.2668
#>  0.1206 -0.1301  0.1674 -0.0375  0.0995
#> -0.0593  0.3456  0.0118  0.1664 -0.0351
#> [ CPUFloatType{10,5} ]
```
