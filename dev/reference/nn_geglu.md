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
#> -0.3219 -0.3356 -0.0686 -0.0184  0.1858
#>  0.0029  0.0383 -1.1513 -0.0697  0.0062
#> -1.5277 -0.1425 -0.0342  0.0143 -0.0259
#>  0.1837  1.1662  0.2174  0.0014 -0.1713
#> -0.0874  0.0667 -0.0249 -0.2433 -0.0098
#> -0.0007  0.0294 -0.0196 -0.5345  0.3256
#> -0.0340  0.4798 -0.0718 -0.3017 -0.0422
#> -1.8227  0.9248  0.1348  0.1765  0.5314
#> -0.0020 -0.0088 -0.1549  0.5024 -0.3328
#>  0.0022  0.0201  0.0476  0.5018 -0.1525
#> [ CPUFloatType{10,5} ]
```
