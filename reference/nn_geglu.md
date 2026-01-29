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
#>  0.0321  0.1101 -0.8474 -0.8986 -0.5025
#> -0.9741  0.0219  0.0256  0.2761 -0.0814
#> -0.4037  0.3452  0.0590 -0.2332 -0.1575
#> -0.0324 -0.0801  0.0186  1.2319 -0.4466
#>  0.0107 -0.2972  0.1383 -0.0660 -0.0343
#>  0.3135 -0.0098  0.2479  0.5673 -0.0747
#> -0.0391  0.0324  0.0740  0.0640  0.2091
#>  0.1838 -2.1114 -0.0269 -0.4890 -0.1297
#> -0.0564 -0.0028  0.0316 -1.1097 -0.0022
#>  0.2671 -0.1633  0.1891  0.2085  0.0940
#> [ CPUFloatType{10,5} ]
```
