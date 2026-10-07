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
#> -0.3243 -0.3685  0.0410 -0.1486 -0.4956
#>  1.7850 -0.1023  0.0333 -0.0018  0.0308
#> -0.0036  0.3946  0.0097  0.0430  0.0097
#>  0.2148 -0.0779  0.3100 -0.1190 -1.6601
#> -0.0335 -0.1533  1.3347 -0.0082  2.4005
#>  0.2136 -1.0287  0.0233 -0.1215 -2.7117
#> -1.2949 -0.0166 -0.1243 -0.1085  0.0190
#> -0.0658 -0.0100 -0.1923  0.6086  0.0468
#>  1.0318  0.0539 -1.5382 -0.1722  0.0513
#> -0.0013 -0.1038  0.2087 -0.0474  0.0004
#> [ CPUFloatType{10,5} ]
```
