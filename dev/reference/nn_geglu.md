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
#> -0.1012  0.5553 -0.0262  0.0035 -0.1148
#> -0.0448  0.1875  0.0071  0.0450 -0.1305
#> -0.1020  2.0066 -0.4164  0.0650 -0.0386
#>  1.7221 -0.0826 -0.0344 -0.0340  0.0852
#> -0.0546 -0.0380 -1.2794 -0.1620  0.9248
#> -0.6319  0.2282 -0.0447  1.3007  0.1343
#> -0.0538 -0.0477  0.8141 -0.0264 -0.0396
#> -0.4691  0.0625  0.2681 -0.4428  0.1108
#> -0.0431  0.0932 -0.1182 -0.0164 -0.1051
#> -0.0977 -0.1861 -0.3456  0.0138 -0.1348
#> [ CPUFloatType{10,5} ]
```
