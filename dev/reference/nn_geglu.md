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
#>  0.0501  0.3426  0.0078  0.6535  0.6674
#>  0.1001  0.0689 -0.1102  0.1491  0.0181
#>  0.4931  0.0951  0.0702 -0.3213 -1.5537
#> -1.8153  0.3111 -0.1377 -0.0119 -0.8594
#> -0.0524  0.0984 -0.1200 -0.0739 -0.0909
#> -0.0159 -0.0387  0.3160 -0.1399 -0.0681
#> -0.0632 -0.4734 -0.8351  0.1500 -0.8081
#> -0.2238  0.1624 -0.0741 -0.0316  0.0430
#>  0.0044  0.2076  0.0150  0.4294  0.0176
#>  0.1074 -0.0285 -0.0072  0.2848 -0.0393
#> [ CPUFloatType{10,5} ]
```
