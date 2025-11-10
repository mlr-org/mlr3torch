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
#> -0.0807 -0.4505  0.2466  0.4991  0.0426
#>  0.0333 -0.1072 -1.5819  0.0326 -0.1087
#>  0.4372  0.2307  0.0029 -0.1647 -0.0276
#> -0.1640 -0.1654 -0.0050  0.2625 -0.1075
#> -0.2948 -0.5829 -0.7907 -0.1754 -0.0781
#> -0.9402 -0.0240 -4.0409  0.8468 -0.2385
#> -1.4794  0.2819  0.2228 -0.0274 -0.0269
#> -0.0133  0.2002 -0.0033  0.0811  0.1351
#> -0.0869 -0.0026  0.0171  0.0728 -0.0013
#> -0.1563  0.0359 -0.0219 -0.0225  0.1184
#> [ CPUFloatType{10,5} ]
```
