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
#> -0.0655  0.2582 -0.2028  0.1616 -0.2820
#> -0.0376  0.0972 -0.0672  0.4027  1.7064
#> -0.1103  0.0507  0.7906 -0.0147  0.0252
#> -0.3903 -1.0912 -0.0985 -0.8466 -0.0452
#> -0.1150 -0.1649  0.0560  2.3590  0.0208
#>  0.5092  0.1784 -0.1743 -0.0188  0.0024
#>  0.0685  0.0022  0.2288  0.0327 -0.1966
#>  0.0163  0.0154  0.2005  0.1436  0.1507
#> -0.4439 -0.0057 -0.7287 -0.0084 -0.2009
#> -0.2098 -0.0852 -0.3026  0.0124 -0.0379
#> [ CPUFloatType{10,5} ]
```
