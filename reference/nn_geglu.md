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
#> -0.4231  0.5941 -0.1570  0.5538  0.0408
#>  0.0801 -0.0116  0.3199  0.0435  0.3263
#>  0.0295  0.0789  3.2826  0.3175 -0.0049
#> -0.0540 -0.0083 -0.0246 -0.0066  0.3325
#>  0.0531  0.0425  0.1566  0.1267 -0.1911
#>  1.9229 -0.1888 -0.0115 -0.0501 -0.1226
#>  0.1273  0.2073  0.0926 -0.8220  0.0411
#> -0.3607 -0.1284  1.3036 -0.0570  0.0572
#>  1.9408 -0.0177 -0.2048  0.0872 -2.2367
#>  0.0400  1.2140  0.0162 -1.2918 -0.0845
#> [ CPUFloatType{10,5} ]
```
