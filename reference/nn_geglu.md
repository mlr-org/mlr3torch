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
#>  0.0291 -0.0495 -0.3583  0.0152  0.0176
#> -0.0480 -0.5419  0.0998  0.0294  0.0570
#>  1.0161 -0.0652  1.7884  0.1347  0.0166
#>  1.5959 -0.0843 -0.0169  0.2160 -0.1839
#> -1.5984  0.0306 -0.1192  0.6546 -0.0741
#>  0.7050  0.0832  0.5323 -0.0262 -1.5737
#>  0.0526  1.5681 -0.0604 -0.0020 -0.0650
#> -0.0466  0.1461  0.0540  0.9824  0.0123
#> -0.0028 -0.4452 -1.8468  0.1185  0.2239
#> -0.1419  0.0429  0.1593 -0.0205 -0.5026
#> [ CPUFloatType{10,5} ]
```
