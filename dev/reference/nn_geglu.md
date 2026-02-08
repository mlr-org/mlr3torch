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
#> -0.4997 -0.0801 -0.0867 -0.8448  0.1022
#>  0.1612  0.0269  0.8787  0.2060  0.0231
#>  0.0377 -0.1402  0.0185  0.3831  0.0984
#>  0.0922 -0.0131 -1.9144  0.0225  0.4780
#>  0.2479 -0.6772  0.8507  0.1065 -0.0211
#> -0.0994 -0.7189 -0.0177  0.0487  0.0495
#>  0.7377 -0.1871 -0.0968  0.1702 -0.0559
#> -0.0259  0.0346  0.1041  0.5601 -0.0475
#>  0.0071 -0.0386 -0.2686  0.0555 -0.1110
#> -0.1315  0.0381 -0.1669 -0.1106  0.0299
#> [ CPUFloatType{10,5} ]
```
