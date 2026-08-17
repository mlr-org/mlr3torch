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
#> -0.2261  0.1363 -0.2125  0.0522  0.0142
#>  0.0680  0.6681  0.1287 -0.0643  0.2263
#> -0.0929  0.0387 -2.4064 -0.0771 -0.0049
#>  0.0695 -0.0269  0.2296 -0.2065 -0.0171
#> -0.0295  0.3959 -2.3798  0.0913  0.0046
#>  0.2101  0.1931  0.1522 -0.1480  0.0867
#>  0.2964  1.4363 -0.0237  0.3632  0.0328
#>  0.0072  0.1645 -0.0065 -2.1107 -0.0057
#>  0.0002  0.1673 -0.0509  3.1998  0.0724
#> -0.0455  0.2309  0.9481 -0.2047 -0.1655
#> [ CPUFloatType{10,5} ]
```
