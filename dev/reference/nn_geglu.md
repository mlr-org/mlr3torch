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
#>  0.0244  0.0727  0.1121  0.0825  0.0659
#> -0.0730 -0.7043 -0.0734 -0.0309 -0.3815
#> -0.2001  0.1786  0.0167 -0.0177  0.0038
#> -0.0032  0.0564  0.0379  1.9165  0.1667
#>  0.0084  0.0681  0.0203  0.0136 -0.0712
#>  1.0307  0.0495  0.1783 -0.0156 -0.0642
#> -0.2544  0.0792  0.0603  0.0179  0.0130
#> -0.1011 -0.0307  0.0406  0.0283  0.4591
#> -0.3049  0.1055  0.6073  0.4935 -0.0217
#>  0.0706  0.0120  0.0905 -0.0500 -1.8054
#> [ CPUFloatType{10,5} ]
```
