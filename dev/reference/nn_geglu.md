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
#> -0.2430  0.8123 -0.0756  0.0242  0.8037
#>  0.0723 -0.0023 -0.3463  0.2428 -0.0054
#>  0.1170  4.3314  0.0217 -0.0722 -0.6502
#> -0.1995 -0.2070  0.0297 -0.2924  0.0714
#> -0.0163  0.2238 -1.2508 -0.0457 -0.0304
#> -0.0105  0.5216 -0.0516  0.0224 -0.1443
#>  0.5118 -0.5160 -0.0681 -0.0358  1.6345
#> -0.4216 -0.2368  0.0109 -0.0389 -0.0079
#>  0.1824  1.1555 -0.5581 -0.0962 -0.0442
#>  0.0494  0.0828  0.0241 -0.1536 -0.0147
#> [ CPUFloatType{10,5} ]
```
