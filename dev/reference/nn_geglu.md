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
#> -0.0185  0.0400 -0.0023  0.3431 -0.0114
#> -0.0003  0.0003 -0.1952  0.0716  0.4522
#> -1.2654 -0.0794  0.3252  2.0117 -0.3036
#>  0.2921 -1.2323  0.0650 -0.6186  0.0109
#>  0.1238 -0.1424  0.2363  0.0867  0.0916
#>  0.0153  0.3427  0.0855 -0.0616 -0.1124
#>  2.4028 -0.0623  0.1568 -0.5678 -0.0683
#> -0.0807  0.0681 -0.1407 -0.3303  0.5739
#>  0.0574 -0.0223  0.0186  0.1723  0.1222
#>  0.0534 -0.0335  0.0250  3.5069 -0.0277
#> [ CPUFloatType{10,5} ]
```
