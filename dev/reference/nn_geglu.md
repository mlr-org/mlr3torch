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
#>  0.0093 -0.0277  0.0822  0.0300  0.0003
#>  0.0127 -0.1610 -0.0016 -0.1276  0.0356
#> -0.0831  0.5433  0.0799 -0.0751  0.0116
#> -0.0815  0.1741  0.1159 -0.4382 -0.0715
#>  0.0182 -0.0957 -1.0869  0.0143 -0.0064
#> -0.0651 -0.9547 -0.0026  0.0158 -0.0030
#> -0.2503 -0.1135  0.0362  0.4502  0.2341
#>  0.1330  0.0464 -0.1720  0.0840 -0.0664
#>  0.2173  0.5520  0.0019  0.6016 -0.2541
#>  0.1416 -0.1615  0.0002  0.2227 -0.0201
#> [ CPUFloatType{10,5} ]
```
