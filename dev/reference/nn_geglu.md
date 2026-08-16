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
#>  0.0074  0.0473 -0.0615  0.3672  0.2285
#> -0.1941 -0.3020  0.0674  0.3085  0.2945
#>  0.0004 -0.0474  0.1208 -0.0158 -0.1383
#> -0.0846 -0.8985 -0.0214  0.1817 -0.1813
#> -0.0094 -0.0069  0.0585  0.7946  0.2077
#> -0.0817  0.2874  0.1479 -0.0879  0.2938
#>  0.7260  0.0779  0.6493 -0.0688  0.0023
#> -0.5028 -0.2986 -0.4405 -0.1225  0.0041
#>  0.1951  0.1036  0.4380 -0.0830  0.0315
#>  0.3274 -0.0587  0.2301 -0.0131 -0.4622
#> [ CPUFloatType{10,5} ]
```
