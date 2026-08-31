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
#> -0.0037  0.4277 -0.1055 -0.0976  0.0254
#> -0.1337 -0.0822  0.0319  0.1975 -0.0412
#>  0.0016 -0.0142 -0.9204 -0.0557 -0.0016
#> -0.0007 -0.0886 -0.0598 -0.2647 -0.0094
#>  0.1955  0.0828  0.2492 -0.3984  0.0098
#>  0.1690 -0.0023 -0.1824 -0.0308 -0.0116
#> -0.0237 -0.2119  0.1528 -0.4403  0.0227
#>  0.0008 -0.2868  0.2419  0.1453  0.0299
#> -0.1019 -0.1069  0.3139 -0.1689  0.3432
#> -0.2051  1.2431  0.8300  2.0244  0.0241
#> [ CPUFloatType{10,5} ]
```
