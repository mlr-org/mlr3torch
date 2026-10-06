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
#>  0.0972 -0.1737  0.0036 -0.0559 -0.9803
#>  0.1074  1.3650  0.4084 -0.0438  0.0240
#> -0.0373 -0.0395  0.1655 -0.6569  0.1602
#>  0.1178 -0.0263  0.1181 -0.1164  0.0134
#>  0.1157  0.6609 -0.0379 -0.5197  0.1420
#> -0.1193 -0.2542  0.1075  2.3643  0.0684
#> -0.0918 -0.0702 -0.1260  0.0958  3.3951
#> -0.0027 -0.2183 -0.3288  0.0424  0.0042
#> -0.0032  0.0681  0.0135 -0.0547  0.3538
#> -1.8380 -0.0464 -0.2215  0.0316  0.6010
#> [ CPUFloatType{10,5} ]
```
