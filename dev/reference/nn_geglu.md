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
#>  0.0048 -0.1404  0.5107 -0.1978  0.1307
#>  0.6509 -0.6470 -0.0145 -0.1637  0.0008
#> -0.7014  0.2121 -0.0075 -0.1537  0.1906
#>  0.0238  0.1092 -0.1094 -0.0100 -0.0674
#> -0.0049  0.0475  0.1161 -0.6468 -0.5959
#> -0.2094  0.0661  0.1426 -0.0296 -0.0936
#>  0.2346 -0.3039  0.3819 -2.3574 -0.0765
#>  0.0008  0.1557  0.0286  0.0191 -0.1202
#>  0.0759  0.0894 -0.0142 -0.0168  0.1184
#>  0.0195 -0.1267  0.4960  0.1223 -0.0200
#> [ CPUFloatType{10,5} ]
```
