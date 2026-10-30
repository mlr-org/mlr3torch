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
#>  0.0000  0.0000  0.0000  0.9321 -0.0000
#>  0.3085  0.1073  0.5029  0.7845 -0.0000
#>  0.6110 -0.0000  0.0038 -0.6926 -0.2135
#>  0.0000  4.5642 -0.0000  0.0000  0.0000
#>  0.0089  0.5215  0.0000 -1.1140  0.0000
#>  0.0000  0.6763 -0.3351 -0.4139 -0.0000
#> -0.0000 -0.8714  0.7165 -0.3817 -0.0000
#>  0.0000 -0.0000 -0.2892  0.0000 -0.0000
#>  0.0000 -0.1499  0.0000 -0.6492 -0.3257
#> -0.0000 -0.1145  0.5926 -0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
