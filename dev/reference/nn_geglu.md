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
#>  0.0318  0.0052 -0.1068  0.0003  0.1240
#> -0.5717  0.0775  0.4406 -0.0473  0.0868
#>  0.2807 -0.3157  0.5176  0.1429  0.0295
#>  0.0248 -0.1370  0.0279 -0.0647  0.9897
#>  0.2688 -0.9964 -0.3947  0.0061  0.2518
#>  0.0709  0.1599 -0.1144 -0.4045  0.0926
#>  0.0540  0.1473  0.6030 -0.0314  0.1216
#>  0.0524  0.1838  0.0039 -0.5229 -0.0864
#>  0.0413  2.8389  0.0446  0.0271  0.0066
#>  0.0300  0.5338  0.1020  0.0753  1.3910
#> [ CPUFloatType{10,5} ]
```
