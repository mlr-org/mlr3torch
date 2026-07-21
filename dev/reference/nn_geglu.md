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
#>  0.0693  0.0040  0.3403 -0.0625 -0.0727
#> -0.0195 -0.1659  0.0007  5.1014 -0.4283
#> -0.1712  0.2820  0.9525 -0.1956  0.2510
#>  0.1002 -0.4804 -0.2755  0.2812  0.3006
#> -0.0445 -0.0616  0.2489  0.3010 -0.0914
#>  0.1373  0.5774 -0.0766  0.0996  0.0002
#> -0.3070 -4.4636  0.0833  0.0038  0.1747
#>  0.4311  0.4227  0.0980  0.1934 -0.0010
#> -0.2309  0.0216  0.1984  0.2353  0.0404
#> -0.0243 -0.6091 -0.2436 -0.1815  0.2726
#> [ CPUFloatType{10,5} ]
```
