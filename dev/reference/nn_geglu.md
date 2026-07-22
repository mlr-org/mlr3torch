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
#> -0.0778 -0.2487 -0.0155 -0.1337 -2.6250
#> -0.0581  0.0380  0.0005 -0.2328  0.1201
#>  0.2492 -0.0496  0.1427  0.5402  0.0379
#> -0.0347  0.3095  0.0781  0.0911 -0.1843
#>  0.1694  0.0715 -0.1233  0.1877  0.9001
#> -0.0856  0.0012  0.3905 -0.0223 -0.0923
#> -0.0743  0.1564 -0.1622 -0.5742  0.0014
#>  0.0518  0.1179  0.1127 -0.0035 -0.1076
#>  0.1773  0.0514 -0.0081 -0.0116 -0.2932
#>  0.2248 -0.2067 -1.2045  0.2558 -0.0387
#> [ CPUFloatType{10,5} ]
```
