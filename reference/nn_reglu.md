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
#>  0.0000 -1.1650  0.0000  0.1184  0.0000
#>  0.0000 -0.3509  0.0000 -0.0000 -0.0000
#> -0.4186  3.0815  0.0000  0.1121 -0.0343
#>  0.3133 -0.1914 -0.0000 -0.0000  0.1564
#>  0.0000 -0.0415  0.0000  0.0000  0.0000
#> -0.0000  0.0000  0.0000 -0.0000  0.0000
#> -0.5128 -0.8126  0.0000 -0.0000  2.8335
#>  0.0000 -0.0000  0.0000  0.0000  1.8693
#> -0.0000 -0.9352  0.0000 -0.2142  0.1590
#> -0.0000  0.0000 -1.7640  0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
