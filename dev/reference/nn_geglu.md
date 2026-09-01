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
#>  0.1183  0.0606 -1.1818 -0.3164  0.9203
#> -0.0327  0.2319  0.2541  0.6027 -0.0314
#> -0.0016 -0.6304 -0.2434 -0.1437  0.7582
#> -0.0070  0.0291 -0.0776 -0.1476 -0.1018
#>  0.2446  0.1575  0.0292  0.0584  0.0114
#>  0.0404  0.6080 -0.0554 -0.1259  0.1478
#> -0.0915 -0.2052  0.2479 -0.0360  0.2487
#> -0.1462 -0.3073 -1.6909  0.1286  0.3161
#> -0.0187  0.0306 -0.4687  0.0454 -0.2282
#> -0.1857 -0.0621  0.0601  2.2355  0.0617
#> [ CPUFloatType{10,5} ]
```
