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
#>  0.0005 -0.0638 -0.5041 -0.0398  1.6312
#>  0.2220 -0.1980  0.1750 -0.1090 -0.6137
#>  0.0351 -0.0242  0.0709 -0.1330 -0.0665
#>  0.0456 -0.3421 -0.3097  0.3443  0.1306
#>  0.0723 -0.0262  1.0455 -0.3906 -0.2454
#> -0.1764  0.0226  0.0099  0.0918  0.0345
#> -0.0709 -0.8711  0.2791 -0.3469  0.0922
#>  0.0035  0.0012  0.0294 -0.0640  0.1186
#> -0.1908 -0.1162  0.0150  0.0073  0.2584
#>  0.6380 -1.3847  1.0728 -4.2365 -0.1333
#> [ CPUFloatType{10,5} ]
```
