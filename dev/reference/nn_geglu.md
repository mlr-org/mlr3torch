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
#>  0.0338 -0.0854  0.1113  0.0363 -0.0298
#>  1.2784 -2.3020 -0.0608 -0.0746 -0.1149
#>  0.0027 -0.0952 -0.1591  0.3128 -0.0754
#>  0.0197  0.0269 -0.1912 -0.4463  0.0693
#>  0.0632 -0.0199  0.8031 -7.7752  0.5573
#> -0.0459  1.1479 -0.0223 -0.0136 -0.1221
#> -0.1134  0.1363  0.1625  0.0676 -0.2480
#>  0.0424 -0.2129 -0.1434  0.0030 -0.0913
#> -0.0688  1.1582 -0.8284  0.1308  0.0745
#> -0.0177  0.8515  1.1179  0.1355  2.6562
#> [ CPUFloatType{10,5} ]
```
