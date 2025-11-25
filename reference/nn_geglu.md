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
#>  0.1102 -0.0792 -0.0378 -0.0279  0.1677
#>  0.0842 -0.8515  0.2586  0.0607 -0.0045
#> -0.1632 -0.8314  0.0131  0.2278  0.1750
#>  0.1572  0.0418 -0.0348  0.7710  1.4809
#>  0.4682 -0.0190 -0.1655  0.0079 -0.1024
#>  2.5206  0.5724 -0.3704 -0.0046  0.4611
#>  0.1173 -0.0856 -0.0152  0.0626  0.3925
#> -0.1228  0.0165 -1.7643 -0.1656  0.2181
#> -0.0716 -0.0744  0.0496 -0.1092  0.0413
#>  0.0550 -2.2095  0.3844  0.3771 -0.5417
#> [ CPUFloatType{10,5} ]
```
