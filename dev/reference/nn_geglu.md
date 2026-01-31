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
#> -0.1554  0.5014  1.7389  0.0001 -0.0950
#> -0.0753  0.0182 -0.0247  0.1143 -0.4862
#> -0.0374 -0.0837  0.1023  0.0186 -0.2203
#>  0.0292 -0.1866 -0.1382  0.5973  0.1593
#> -0.0143 -0.0335  0.0878 -0.0647 -2.6359
#> -0.0044 -0.1422  0.0522 -0.1659  0.0338
#> -0.0041 -0.0035 -0.1598  0.5447  0.1663
#> -0.1736  0.1160  0.7250 -0.0359  0.0715
#> -0.2106 -0.0147 -0.1094  0.0641 -0.1564
#> -0.0004  0.0297  0.1129  0.1661  0.0827
#> [ CPUFloatType{10,5} ]
```
