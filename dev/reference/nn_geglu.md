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
#>  0.0459 -0.3176 -1.0132 -0.0170  0.2471
#> -0.1258  0.0506 -0.0667  0.0210 -0.2701
#>  0.0768 -0.1742  1.9455  0.1542 -0.0634
#> -0.0299 -1.8483  0.1851 -0.2186 -0.0725
#>  0.0979  0.5273  0.2708  0.1219  0.0839
#>  0.0973  0.0905  0.0019 -0.1310 -0.0344
#> -0.4612 -0.0616 -0.0002 -0.0986  0.1048
#> -0.1841 -0.0777  0.0248  0.0035  0.0016
#> -0.2550 -0.0527  1.4577 -0.2639  0.3247
#>  3.1665  0.1947 -0.3103  0.1268 -0.1143
#> [ CPUFloatType{10,5} ]
```
