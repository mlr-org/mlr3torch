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
#> -0.0586  0.1055 -0.0522  0.0485  0.5117
#> -0.4564  0.2028  0.0972 -0.2339 -0.0167
#> -0.0258 -1.2177 -0.1025  1.5063 -0.0040
#>  0.0587  0.2994  0.1647 -0.0563  0.1903
#>  0.0184  0.3346 -0.0208  0.0002  0.2875
#>  0.2421 -0.1018 -0.3628 -0.0777  0.1231
#>  0.0247 -0.0741  0.0010 -1.2993  0.2739
#>  0.1003 -1.3302  0.1077  0.2173 -0.0191
#>  0.0399 -0.4771 -0.0234 -0.6601 -0.0751
#> -0.9370 -0.0200 -0.0107  0.0451  0.0514
#> [ CPUFloatType{10,5} ]
```
