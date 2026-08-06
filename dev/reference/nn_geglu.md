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
#> -0.1711 -0.2491  0.1176 -0.1230 -0.1203
#> -1.3311 -0.8785 -0.0551 -0.0996  0.1443
#>  0.0265 -0.0387  0.2562  0.8606 -0.8141
#> -0.0387 -0.0221  0.1093 -0.0485 -0.4993
#> -0.0686  0.0699  0.8872  0.0245 -0.1036
#> -0.9344  0.3375  0.0928  0.1345  1.6164
#>  0.0959 -0.0061 -0.0283  0.0003 -0.0309
#>  0.0128  0.0835  0.2224  0.0495 -0.1318
#> -0.1937 -0.0136  0.1235 -0.0817 -0.0024
#>  0.1048  0.1105 -0.1398  0.0717  0.2175
#> [ CPUFloatType{10,5} ]
```
