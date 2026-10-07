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
#>  3.1594  0.4723  0.0092  0.0576  1.7554
#>  0.4736  0.0056 -0.0807  0.9609 -1.1299
#> -0.0360  0.0040 -0.0504  0.5861 -0.1135
#> -0.0019 -0.1773 -0.0214 -0.1265 -0.1401
#>  0.0851  0.0173  0.1053 -0.1239 -0.1705
#> -0.0286 -0.0465 -4.4750 -0.3059  0.0841
#> -0.0312 -0.1034 -0.0113 -0.2696  0.0099
#>  0.4734  0.0229 -0.0066 -0.1195 -0.0081
#>  0.0636  0.1635 -0.0644  0.3666 -0.3182
#> -0.2333 -0.1026 -0.0472 -0.0816 -0.9276
#> [ CPUFloatType{10,5} ]
```
