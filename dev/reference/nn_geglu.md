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
#> -0.0009 -0.1510 -0.0013  0.0557 -0.1218
#> -0.0404  0.0197  0.1853 -0.4874  0.1154
#>  1.3464 -0.0666 -0.0600 -0.2871  0.0654
#>  0.0517 -0.1966 -0.0090 -0.1328  0.0822
#>  0.0306 -0.0121  0.6841  0.2444  0.0385
#> -0.4654 -0.0967 -0.2441  0.0456 -0.2731
#>  0.6236  0.0140  3.1313 -0.1608  0.1369
#> -1.0052  0.0760  0.7499 -0.1378 -3.6723
#>  0.0112 -2.8868 -0.2267 -0.1864  1.0440
#> -0.1476 -0.0046 -0.0016 -0.1881 -0.0062
#> [ CPUFloatType{10,5} ]
```
