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
#> -0.2242 -0.1431  0.0781  0.3105 -0.0087
#>  0.1466  0.6187  0.6413  0.5589  2.0860
#>  0.3256 -0.0076 -0.1629  0.7344  0.3780
#> -0.7074  1.1923 -0.1494  0.2643  0.1377
#> -0.3663 -0.0593 -0.0238 -0.0247 -0.4489
#>  0.1308 -0.0569 -0.4974  0.7549  0.0517
#> -0.0218  0.0088 -0.5075 -0.1310 -0.0115
#>  0.0382  0.1893 -0.3315 -0.2324  0.0668
#> -0.1060  0.0167  0.0280 -0.0799  0.0354
#> -0.0026  0.0148  0.1358 -0.0581 -0.0550
#> [ CPUFloatType{10,5} ]
```
