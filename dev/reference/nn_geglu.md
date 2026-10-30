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
#>  0.0117  0.2074 -1.5336  0.0721  0.0494
#>  0.0507  0.0828 -0.3591  0.8811  0.0447
#>  0.1718 -0.0185 -0.1357 -0.1982 -0.3193
#> -0.9579  0.1385 -0.2943 -0.1440  0.1779
#>  0.5182 -0.0398  0.0014 -0.0256  0.4114
#>  0.1751 -0.0348 -0.0156  0.0168  2.9445
#>  2.5785  0.2187  0.0372 -0.2399  0.1446
#>  0.2101  0.2539  0.0105 -1.0929  0.3815
#> -0.0835  0.2318  0.1153 -0.0055 -0.0020
#> -0.0568  0.0652  0.3532  0.0184 -0.0371
#> [ CPUFloatType{10,5} ]
```
