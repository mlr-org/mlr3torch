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
#> -0.1439  0.0000  0.0150  1.2925 -0.2475
#>  0.0873  0.0241  0.0000  0.4201  0.0000
#>  0.6731 -0.6129 -0.0000 -0.0000 -0.0000
#>  0.0000  1.5607  0.0000  0.1559  1.7757
#> -0.1267 -0.0000  0.2445 -0.0000  0.0000
#> -0.1506 -0.0000 -0.0000 -0.0000  1.6036
#> -0.3110  0.0000 -0.1442 -0.0840 -0.0000
#>  0.3410  0.0000 -1.2572 -0.0000  0.8448
#> -0.0000  0.0000  0.2563  0.0000  0.6144
#>  0.0000  0.0000 -0.0000  0.0000  0.7720
#> [ CPUFloatType{10,5} ]
```
