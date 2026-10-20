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
#> -0.0201  2.6137  0.0764  0.2705 -0.0648
#>  0.1792 -0.0036  0.0666  0.0288  1.2123
#>  0.0241 -0.0006 -0.2111 -0.0473  0.0063
#>  0.0346  0.1098  0.4881 -0.0064 -0.0413
#> -0.1024 -0.5093  0.4410 -0.8940  0.1403
#>  0.0819  0.1806  0.0243 -0.2074 -1.1308
#>  0.6629 -0.0346 -0.2223 -0.3002  0.1369
#>  1.0389 -0.1896 -0.0270 -0.1117  0.0562
#>  0.1346  0.0570  0.0857 -0.1134 -1.1227
#>  0.6400  0.1864 -0.0191  0.0526  0.2066
#> [ CPUFloatType{10,5} ]
```
