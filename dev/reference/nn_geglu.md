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
#> -0.0108 -0.0473 -2.8381  0.1869  0.0209
#>  0.0657 -0.0556  0.0009  0.2142 -0.1355
#> -0.1896  0.2050 -0.0274 -0.6712  0.1545
#>  0.0476  0.3556  1.6048 -0.1181  0.0181
#> -0.0058 -0.2241  0.0502  0.1507 -0.3133
#> -0.0454 -0.1085 -0.0780 -0.1345 -0.2340
#>  1.2593 -0.0418 -0.0553 -0.1683  0.1014
#> -0.1666  0.4040  0.6136 -0.1083  0.0216
#> -0.0636 -0.2046  0.1259 -0.0874  0.5012
#>  6.7637 -0.0179  0.6726 -0.2151 -0.1313
#> [ CPUFloatType{10,5} ]
```
