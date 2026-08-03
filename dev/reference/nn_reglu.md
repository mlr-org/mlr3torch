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
#> -0.0000  0.0000  0.0000 -0.2210 -0.4453
#>  0.1308 -0.2387  0.0000 -0.0000 -0.0000
#>  0.0000  0.0071 -0.0000 -0.0000 -0.0123
#> -0.0000  0.1560  0.3453 -0.0000  0.0000
#> -0.0000 -0.0000  0.0000 -0.0000  1.1351
#>  0.1567  0.1475  0.0000 -0.0000 -0.2062
#> -0.7031  3.4365  0.0000  0.6169  0.0000
#>  0.0000  0.0000 -0.6243  0.0000 -1.0657
#> -0.4385 -0.0000  0.0000 -0.0000  3.0869
#> -0.0000 -0.2163 -0.7151 -0.2594  0.0000
#> [ CPUFloatType{10,5} ]
```
