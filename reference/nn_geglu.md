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
#>  2.3116 -0.9265  0.2237  0.2567  0.0306
#>  1.2897  0.1652  0.3782 -0.0417 -0.1458
#> -0.0695 -0.2003  0.0574  0.0761 -0.0084
#>  0.5929  0.0703 -0.2193 -0.3007 -0.0745
#> -0.1022 -0.2176  0.1893 -0.7265  0.3330
#>  0.0170 -0.1259 -0.0240  0.1559  0.1187
#> -1.8190  0.8020  1.2051  0.0969 -0.0516
#> -0.2443 -0.1667 -0.0759  0.1780 -0.3374
#> -0.0638  0.0525  2.5270 -0.0523 -1.5847
#> -0.2631  0.0649  0.2098  0.0245  1.0449
#> [ CPUFloatType{10,5} ]
```
