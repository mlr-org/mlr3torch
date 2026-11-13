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
#> -0.0000  0.0000  0.0000 -0.0000  0.6334
#>  0.0000 -0.0000 -0.0000 -0.0000  0.2425
#>  2.2751  0.0000 -0.0000  0.0000  0.0000
#> -0.1152  0.0004  0.1835  0.0000  1.2869
#>  0.0000 -0.0000 -0.0000  0.0000  1.6513
#> -0.0000 -0.0969 -0.0000 -0.0114  0.0000
#> -0.0223 -0.0000 -0.7150 -0.0000  0.0051
#> -0.0566 -0.9137 -3.0281  0.3216  0.0000
#>  0.0632 -0.0000 -0.3434  0.0000 -0.0000
#>  0.0000 -0.0000 -0.0000  1.2967  0.0000
#> [ CPUFloatType{10,5} ]
```
