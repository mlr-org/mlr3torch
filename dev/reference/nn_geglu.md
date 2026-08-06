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
#>  0.0033 -0.3491  0.0378  0.0429  0.0569
#> -0.0698  0.2605 -0.0703  0.3129  0.0640
#>  0.1532  0.0818  0.1434 -0.0061  0.0187
#>  0.2202 -1.5543 -0.0033  0.6760 -0.2602
#> -0.2711  0.0326  0.1967  0.2815  0.8256
#>  0.5120 -0.8313 -0.0941 -2.5757  0.4453
#> -0.0077 -2.4112 -0.0004 -0.1317  0.1900
#>  0.4782 -0.3445 -0.0684 -1.3235  0.0230
#> -0.0133  1.0960 -0.0145  0.0049 -1.6009
#>  2.0893  0.0101  0.0128 -0.1853 -0.0817
#> [ CPUFloatType{10,5} ]
```
