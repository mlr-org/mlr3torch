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
#> -0.3573 -0.1997 -0.0239  0.3885  0.1521
#>  0.6197 -1.5779 -0.0064  0.1549  0.0054
#> -0.0080  0.6279  0.0921  0.4125 -0.3494
#> -0.0632 -0.1135 -0.2468  0.0427  0.1319
#> -0.1217  0.8083 -0.9877  0.1304 -0.1571
#>  0.0237 -0.0155  0.2786  0.0367  0.0173
#>  0.2357 -0.8032 -0.0811  0.0941  0.0720
#> -0.0044  0.0072 -0.0997  0.0355  0.0220
#>  0.0104 -1.3411  0.0065  0.3470 -1.7118
#>  0.0144 -0.2945 -0.0973  0.0025  0.0335
#> [ CPUFloatType{10,5} ]
```
