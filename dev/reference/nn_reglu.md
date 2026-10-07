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
#> -0.8637  0.0000  0.6988  0.4318  0.4116
#> -0.0000  0.0000  0.0000 -2.0046 -2.8483
#>  0.0108 -0.0000  0.0000 -0.0754 -2.2978
#>  0.0000 -0.0000  0.2013  0.0000  0.1263
#> -0.0000  0.0000 -0.4415  0.2129  0.0000
#> -0.0000  0.0575  0.0228  0.0072  0.0000
#>  0.0000 -1.5832  0.0000 -0.0000  0.0869
#> -0.0000  0.0000  0.2508 -0.0359 -0.0000
#> -0.4170 -0.2507 -0.0000 -0.0000  0.0000
#>  0.0000  0.0806 -0.0902  0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
