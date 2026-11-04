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
#> -0.2205 -0.0649  0.7116 -0.0953  0.1876
#>  0.0200  0.1308  0.0101  0.6935 -0.9040
#>  0.1537  0.4104 -0.4193 -1.4024 -0.0555
#> -0.0202 -0.0570 -0.0940  0.0726  1.2262
#>  0.4468 -0.1467  0.3146  0.0268 -0.2909
#>  0.0508 -0.0779  0.1855  0.0358  0.0458
#>  0.1838  0.0031  0.0075 -0.0680 -0.0151
#>  0.2336  0.8853 -0.0445 -0.3512  1.6410
#>  0.1066 -0.0509 -0.3021 -0.3037  0.1307
#>  0.0032  0.3165 -0.9477  0.1426 -0.7852
#> [ CPUFloatType{10,5} ]
```
