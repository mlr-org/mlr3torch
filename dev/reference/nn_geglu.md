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
#> -0.5025  0.2964  0.0733  0.3499 -0.1957
#> -0.0847  0.8721 -0.0070  0.9425  0.0229
#>  0.3038  0.1070  0.1514  1.5181  0.1013
#> -0.5783 -0.0846 -0.0221 -0.0119 -0.2913
#> -1.1238 -0.2160 -0.0741 -0.0849  0.1914
#>  0.4026  0.2041 -0.1044 -0.0359 -0.0537
#>  0.0108  0.0510  0.0804 -0.3159  0.1351
#> -0.0561  0.1380 -1.6806 -0.0508 -0.1848
#> -0.2487 -0.1294  0.1918  0.3614  0.2611
#> -0.0902 -0.2266 -0.1174 -0.1135 -0.0556
#> [ CPUFloatType{10,5} ]
```
