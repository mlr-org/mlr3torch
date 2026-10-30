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
#> -0.0176 -0.0895  0.4257  0.2416 -0.5798
#>  0.0364 -0.0035 -0.0706  0.5959 -0.3941
#> -0.0445  0.0158 -0.0421  0.0508  0.1009
#> -0.3379  0.0477 -0.1166 -0.7121 -0.6918
#>  0.2938  0.1055  0.6050 -0.1429  0.7048
#>  0.0382 -2.7664 -0.6462  0.0009  0.0978
#> -0.1027 -0.0231  0.0408  0.6769  0.0962
#> -0.1030 -0.1101  0.9319  2.2631  0.6203
#> -1.3094  0.1874 -0.0386 -0.4128 -0.4456
#> -0.0062 -0.6159 -0.1599  1.5619 -0.0235
#> [ CPUFloatType{10,5} ]
```
