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
#>  0.0000 -0.9924 -0.0000  0.0000 -0.0000
#>  0.7249  0.1384  0.0000  0.7567 -0.4498
#>  0.0329  0.1335 -0.1026  0.0187 -0.4236
#> -0.0000 -0.0454  0.0000  0.0000 -0.0000
#>  0.1592  0.0000 -0.0000 -0.2594  0.0319
#>  0.0000 -0.2299 -0.0000  1.2012 -0.0000
#> -0.0000  0.7152 -0.2556  0.1301  0.0000
#> -0.0293 -1.3190  0.9569 -1.2585  0.0000
#> -0.0760  0.0000 -0.0795 -0.0000 -0.0000
#>  0.4921  0.0511 -0.0000  0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
