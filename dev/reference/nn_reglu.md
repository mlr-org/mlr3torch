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
#>  0.6963 -0.2316 -0.0000  0.0000  1.2998
#>  0.0000 -0.0000 -0.0000  0.0000 -0.0000
#> -0.0000  0.0000 -0.0000  0.6532  0.0000
#>  0.1051 -0.0000 -0.0000 -0.3324  0.0000
#> -0.0000  0.0000  0.0000 -0.0000  0.0000
#>  0.0043  0.0000  0.0000 -0.0000  0.0000
#>  0.0311 -0.0000  0.0000  0.3736 -0.0000
#> -0.1163 -0.0000  0.2469  0.0000  0.0000
#>  0.0313 -0.6313  0.8854  0.0000 -0.0145
#> -0.0000 -0.0000 -0.0000 -0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
