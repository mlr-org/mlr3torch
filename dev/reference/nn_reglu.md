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
#> -0.0101 -0.5535 -0.0000  0.0000 -0.0000
#>  0.0000  1.3270 -0.0000  0.0000  0.0113
#> -0.0000  0.0818  0.0000 -0.1713  0.0197
#> -3.7102  0.0000 -0.0000  0.0000  0.0000
#> -0.0000  0.3208 -0.0000  1.2474  0.0000
#> -0.5419  1.3796  1.4084  0.2557  0.0000
#>  0.4155 -0.7262 -0.0000  0.1376 -0.0184
#> -1.0576  0.1694  0.0000  0.0000 -0.1365
#> -1.0218 -0.0000  0.4722  0.0278  0.0000
#>  0.0355 -0.8125 -0.0000 -0.5035  0.0000
#> [ CPUFloatType{10,5} ]
```
