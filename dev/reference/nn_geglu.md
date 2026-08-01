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
#> -0.0041 -0.0983 -0.3044 -0.2564 -0.6908
#> -0.0534  0.1495 -0.0779  0.0489 -0.0600
#> -0.2898 -0.0982  0.0803 -0.0503  0.1114
#> -0.0192 -0.5079  0.2498  0.0870 -0.1154
#> -0.0685  0.1257  0.9589  0.1091  0.1172
#> -0.0512 -0.0284  0.2525 -0.1005  0.1476
#>  0.0968  0.0127 -0.1357  0.3382 -1.1686
#>  0.0013  0.0007  0.6153 -0.5744 -0.0733
#> -0.0217  0.0464  0.1109  0.0210  0.1210
#> -0.0225 -0.0395 -0.0038  0.1061 -0.0152
#> [ CPUFloatType{10,5} ]
```
