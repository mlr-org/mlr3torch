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
#> -0.0000 -0.0000  0.0000 -0.5338  0.0000
#> -0.0690  0.0000 -0.0000  0.0000  0.0000
#>  0.1782  0.0000 -1.6053  0.0000 -0.0137
#>  0.9638 -0.1036  0.0000 -3.5718 -0.7352
#>  0.0605 -0.0000  0.0121 -0.0000 -0.0000
#> -0.9677 -0.0000 -0.1805  0.2262 -0.0000
#> -0.9041  0.0000  0.0000  0.0000  0.0014
#> -0.0814 -0.0000 -0.0000  0.4086  0.1027
#>  0.1013  1.7749 -0.0000  0.0000 -0.2312
#>  0.0000 -0.1580  0.0000 -0.3755 -0.0024
#> [ CPUFloatType{10,5} ]
```
