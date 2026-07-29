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
#> -0.0864  0.1155 -0.0063  0.6779  0.0103
#>  0.0006  0.0517 -1.9252 -0.0705  0.1629
#> -0.1243  0.4915 -0.0630  0.0289 -0.9544
#> -0.1223  0.1521  0.1165  0.6353  0.1472
#>  0.0045 -0.4550  0.0226  0.0518  0.9527
#>  0.7370  0.0079 -0.0142  0.0969  0.0484
#>  0.0488 -0.2572  0.0175  0.0378  2.1283
#> -0.1510  0.3112 -0.3348 -0.0018 -0.4085
#>  0.1974  0.2216  0.0365  1.4536 -1.7698
#> -0.0185  0.0048  0.0259  0.1049 -0.0396
#> [ CPUFloatType{10,5} ]
```
