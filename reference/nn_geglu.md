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
#>  0.0225 -1.6916 -0.1269 -0.1086  0.0633
#> -0.0400  0.2295 -0.2154  1.1475  0.2695
#>  0.0143  1.9887  0.0275 -0.1112  0.3402
#>  0.0666 -0.4155 -0.4141 -0.0379 -1.0903
#>  0.0435  0.3687 -3.4553 -0.1652  0.0464
#>  0.0779 -0.0093  0.0790  0.2557 -0.0580
#> -0.1097  0.8459  0.0006 -0.0103 -0.2697
#> -0.0523 -0.2135 -0.0003 -0.1717  0.2807
#> -0.0588  0.1451 -2.8760  0.0671 -0.1542
#> -0.2784 -0.0864  0.1003  0.0609 -0.0167
#> [ CPUFloatType{10,5} ]
```
