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
#>  0.2816 -0.0879  1.8929  0.0651 -1.0704
#>  0.2124  0.3471 -0.0961  0.2088 -0.1375
#> -0.0649 -0.1301  0.0517  0.0835  0.0647
#> -0.3103 -0.0202  0.5162 -0.0900 -0.0077
#>  0.3153  0.0391  0.0790 -0.4879  0.0281
#> -0.3862  0.0052 -0.0358 -0.4227  0.2722
#>  0.0659  0.5589 -0.0234 -0.0819  0.0165
#>  0.0109  0.1512  0.2295  0.1893 -0.0081
#> -0.0571 -0.0870  0.1412 -0.1728 -0.2709
#>  0.9314  0.1087 -0.1345  0.0826  0.1015
#> [ CPUFloatType{10,5} ]
```
