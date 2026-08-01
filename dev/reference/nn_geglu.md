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
#>  0.0105  0.2529  0.0175  0.5593  0.0604
#>  5.4436  0.3302  0.1401 -1.2327 -0.1843
#>  0.0020 -0.0097 -0.0039  1.1041 -0.1562
#>  1.8475 -0.2992  0.1010 -0.0079 -0.0270
#>  0.0608  0.1504 -0.0548  0.2529  0.0290
#> -0.0854  0.0147  0.4820  0.0118 -0.0006
#>  0.4672 -0.4784  0.0113  0.3136 -0.0371
#>  0.0713  0.3378  1.1658  1.7299  0.2330
#> -0.9822 -0.0999  0.0294  0.1358 -0.0347
#> -0.0172  0.0715 -0.2188 -0.0050  0.2816
#> [ CPUFloatType{10,5} ]
```
