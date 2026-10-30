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
#> -0.0766  0.0524  2.0194  0.5330 -0.0917
#> -0.0691  0.4418 -0.1320 -0.2068  2.2332
#>  0.0902 -0.0048 -0.6987  0.3160  0.1169
#> -2.2603  0.0518 -0.0503  0.0297 -0.0103
#>  0.0560 -0.0477  0.0254 -0.0794  0.0336
#>  0.0170  0.0553 -0.0613  0.4294  0.1332
#>  0.0989 -0.3140  0.0811  0.1051  0.1778
#> -0.4164  0.2112  0.7179 -0.6816 -0.3337
#>  1.2085 -0.0263 -0.2292 -0.0002  1.3591
#> -0.2827  1.7093 -0.0347 -2.4333 -0.0032
#> [ CPUFloatType{10,5} ]
```
