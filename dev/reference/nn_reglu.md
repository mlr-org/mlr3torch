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
#> -0.0000 -0.2743 -0.1185 -0.0000 -2.0692
#>  0.0000  1.3170  0.0681  0.0000  0.0000
#> -0.0672 -0.0000 -0.0758 -0.4837 -1.5109
#>  0.0000 -0.0000 -0.0000  0.0000  0.0000
#>  0.0000 -0.0000  0.0000  0.2133 -0.0000
#>  0.9788 -0.5681 -0.0000 -0.0552 -1.0547
#> -0.8479  0.2552 -0.0000  0.0000 -0.4432
#>  0.0000  0.0000  0.0000 -0.0000 -0.0000
#>  0.0101  0.0000  0.1791 -0.0000  0.3868
#> -0.0000 -0.2312  0.0000 -0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
