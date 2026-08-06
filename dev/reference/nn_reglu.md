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
#> -0.0000 -0.0000  3.3688  0.0000 -0.2365
#>  0.0000  0.0000 -0.0000  0.6585  0.3763
#>  0.0000 -0.0000 -0.0000  0.2405  0.0000
#> -0.0000 -0.2208  1.0312  0.0000 -0.6476
#> -3.3220  0.0000 -0.7832  1.2910 -0.9316
#>  1.4375 -1.3389 -0.4089  1.3483  0.0000
#>  0.0000 -0.0000 -0.0000  0.3802  0.0000
#>  0.0000  0.0000  0.0000 -0.0000 -0.0000
#>  0.0533  0.0683  0.0000  0.0000  0.0000
#>  0.0000  0.6254  0.1069  1.0034 -0.1615
#> [ CPUFloatType{10,5} ]
```
