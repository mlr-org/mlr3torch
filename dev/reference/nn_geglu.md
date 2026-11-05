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
#> -0.0455 -0.1190 -1.1763  0.0502  0.1526
#> -0.2621  0.1216  0.7018  0.1018 -0.1563
#> -0.0410 -0.2300  0.3595 -0.8174 -0.0597
#>  0.0390  1.0680 -0.9436  0.2779 -0.3389
#>  0.1761  0.2141  0.2082 -0.0874 -0.0149
#>  0.0027 -0.0142 -0.2266  0.1420  0.2225
#> -0.0241 -0.0393 -0.1637 -0.0694  0.0103
#> -1.8670  0.1074  0.0835 -0.0175  0.5820
#>  0.1986  0.2258 -1.1828  0.0185 -0.2531
#>  0.4844  0.0844  0.0761 -0.0217 -0.1304
#> [ CPUFloatType{10,5} ]
```
