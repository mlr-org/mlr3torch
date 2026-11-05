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
#> -0.6492 -0.0581 -0.0267 -0.9870  1.4205
#>  0.0123 -0.4025  0.0438  0.0317 -1.7285
#> -0.9812  0.5864 -0.0009  0.0193  0.0578
#> -1.4076  0.0319 -0.0334  0.0640  0.0732
#> -0.0424 -0.3001  0.0162  0.0638  1.9584
#> -0.1892  0.0216  0.0490  0.1186  0.1530
#> -0.0754 -0.0004 -0.0188 -0.7205  0.7618
#> -0.1031  0.5250  0.8080 -0.1177  0.2449
#>  0.5283  0.1101  0.0900  0.2160  1.4924
#>  0.0182 -0.3821  0.2001  0.1376 -0.0766
#> [ CPUFloatType{10,5} ]
```
