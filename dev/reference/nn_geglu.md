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
#>  1.6692  0.0511 -0.0984 -0.2093  0.0202
#> -0.0529 -0.0983  0.1660  0.0652  0.3688
#>  0.7708 -0.3117  0.0018  0.2706  0.2737
#>  0.0646 -0.1838  0.1113  0.0760 -0.0048
#>  0.3488 -0.0763  0.0171  0.2631 -0.0013
#> -0.1078  0.0042 -0.0330  0.0305 -0.0692
#>  0.1022  0.0969 -0.0099 -0.0230 -1.0155
#> -0.7642  0.5712  0.0356 -0.0246 -0.2358
#>  0.4433  0.1617 -0.0343  0.1967 -0.0823
#> -0.2053 -1.7506  0.2221  0.0136 -0.0090
#> [ CPUFloatType{10,5} ]
```
