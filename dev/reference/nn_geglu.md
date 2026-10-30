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
#>  0.8479  0.2477 -0.7440  0.0329 -0.6331
#>  0.2353 -0.1391 -0.0748  5.2356 -0.1368
#> -0.0103  0.0560 -0.0724 -0.0465 -0.0886
#> -0.1320  0.0677 -1.4116  0.0758 -0.5750
#> -0.2026  3.1389 -0.3064 -1.7915 -0.1290
#> -0.1754 -0.2661  1.0635 -0.0371  0.2049
#> -0.3307 -0.5438 -0.0775  0.0740  0.0217
#>  0.0465  0.1674 -0.0270  0.0745 -0.8226
#> -0.2600  0.0754  0.6662 -0.0258  0.0170
#> -0.0106  0.2127  0.0344 -0.0491 -0.2808
#> [ CPUFloatType{10,5} ]
```
