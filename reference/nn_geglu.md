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
#> -0.6448  1.2544  0.0303  0.0074  0.0347
#>  0.0943  0.0568 -0.1907  0.0089  0.0670
#> -0.1975 -0.0498 -0.0022 -0.0538 -0.1187
#> -0.0574 -0.1001 -0.1260 -0.0590  0.0575
#> -0.0208  0.2990  0.1109 -0.0422 -0.1352
#>  0.3155 -0.2870 -0.1622 -0.4994  0.1790
#>  0.0242 -0.2345  0.1384 -0.1775  0.0346
#>  0.1019  0.1163 -0.0736  0.1472  0.0025
#> -0.2399  0.1738 -0.0071  1.9775 -0.0463
#> -0.0546 -0.0936 -0.2813 -0.2725 -0.6555
#> [ CPUFloatType{10,5} ]
```
