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
#>  0.7537  0.0363 -0.1896  0.0000 -0.0042
#> -0.0575 -3.0910 -0.0764 -0.1988  0.0000
#> -1.1692  0.0000 -1.8763  1.6819 -0.1925
#>  0.5272  1.2847  0.8080 -0.9785 -0.0000
#> -0.3315  0.0000 -0.4162  0.0000 -0.0153
#> -0.7298  0.0000 -0.1006 -0.0006 -0.3705
#> -0.0000  0.5736 -0.0000  0.0148  0.0000
#>  0.0264  0.0000 -0.0000  0.0000  0.0000
#> -0.2324  0.1147 -1.5172 -0.0000 -0.0000
#>  1.5572 -0.3008  0.0000  0.6872 -0.7599
#> [ CPUFloatType{10,5} ]
```
