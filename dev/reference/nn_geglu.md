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
#>  0.0695 -0.0319 -1.2184 -0.0629  0.0067
#>  0.0282  0.1313 -0.0298 -0.0349  0.0430
#>  0.1095  0.5856  0.1042 -0.0688  0.4845
#>  0.2694 -0.2359 -0.1165 -0.6779 -0.0343
#> -0.0070 -0.0689 -1.4838 -0.1300  0.0435
#> -0.1611 -0.0203  0.0247  0.1570 -0.0051
#> -0.0954  0.0882  0.0828  0.0913 -0.0653
#> -0.0004  0.0147  0.0409 -0.0328 -0.2376
#> -0.1948 -0.4699 -0.2306  0.2783 -1.1979
#> -0.2204  1.9081 -0.2392 -0.0434  0.4435
#> [ CPUFloatType{10,5} ]
```
