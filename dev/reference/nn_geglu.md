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
#> -0.7218 -0.2612 -0.2348 -0.0429 -0.0257
#>  0.1783 -0.0031  0.1133  0.0142 -0.1798
#> -3.6060  0.1334  0.0639 -0.0439  0.0028
#> -0.0699 -0.1353  0.3869  0.1895  2.1974
#> -0.0953 -0.0199 -0.0125 -0.1699 -0.9624
#> -0.1096  0.0293 -0.6391  0.4721 -0.2451
#> -0.0134 -0.2362  0.0223  0.1912 -0.0775
#> -0.1508 -0.1139  0.1694 -0.4395  0.1112
#>  0.5068  0.0905  0.3270 -0.0187 -0.0072
#> -0.0224 -0.0834 -0.2117  0.7157 -0.1771
#> [ CPUFloatType{10,5} ]
```
