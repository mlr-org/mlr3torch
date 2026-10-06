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
#> -0.0306  0.5544  0.0000 -0.0000  0.0000
#> -0.0973 -0.6652  2.7245 -0.0000 -0.5605
#>  0.5128 -0.0000 -0.0002  0.2163 -0.5884
#>  0.0000 -0.0000  0.3622 -0.0000  0.6103
#> -0.0000  0.0000 -0.1462  0.0000  0.1137
#>  0.0000 -0.0000  0.9208  0.8232 -0.2105
#> -0.0000  0.0281 -0.0000  0.0000 -0.6830
#>  0.3436 -0.0000  0.2910  0.0000  0.0000
#> -0.1194 -0.1423 -0.0000  0.8187  0.0288
#>  0.0124 -0.0000 -0.2498 -0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
