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
#> -0.0003  0.1298 -0.3838 -0.0863 -0.0831
#>  0.0735  0.0529  0.0709  0.0101 -0.0463
#>  0.0011 -0.0920  0.0007 -0.0359 -1.5956
#> -0.0291  0.0535 -0.0184  0.5393  0.0095
#>  0.0072 -0.2572 -0.8131  0.7243  0.1669
#>  0.1528  0.0060  0.0910  0.0490  1.2860
#> -0.6572  0.0139  0.2886 -0.2978  0.0916
#> -0.0223 -0.1852  0.0852  0.0878  0.0499
#> -0.0962 -0.3475 -0.0159 -0.1547 -0.0156
#>  0.9758  0.0038 -0.9002  0.3576 -0.0016
#> [ CPUFloatType{10,5} ]
```
