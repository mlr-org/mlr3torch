# ReGLU Module

Rectified Gated Linear Unit (ReGLU) module. Computes the output as
\\\text{ReGLU}(x, g) = x \cdot \text{ReLU}(g)\\ where \\x\\ and \\g\\
are created by splitting the input tensor in half along the last
dimension.

## Usage

``` r
nn_reglu()
```

## References

Shazeer N (2020). “GLU Variants Improve Transformer.” 2002.05202,
<https://arxiv.org/abs/2002.05202>.

## Examples

``` r
x = torch::torch_randn(10, 10)
reglu = nn_reglu()
reglu(x)
#> torch_tensor
#> -0.0000  0.0000  0.0000 -0.8364  0.0000
#>  0.0184  0.2001  0.0528 -0.0000  0.0000
#> -0.0000  0.1915  1.0774 -0.0000  0.0000
#> -0.0000  0.0506  0.7096  0.0000  0.0000
#>  0.0000 -0.0000 -0.0000  0.0000  0.1975
#>  0.0000 -0.0000 -0.0000  0.3025  0.0000
#>  0.0000  0.6703 -0.0000  1.0341 -0.0000
#> -0.0000 -0.4093 -0.0000  0.0296  0.0000
#>  0.1369  0.1162 -0.0000 -0.2362 -0.0000
#>  0.1189 -0.0000  0.2104  0.0000 -0.9215
#> [ CPUFloatType{10,5} ]
```
