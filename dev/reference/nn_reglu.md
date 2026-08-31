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
#>  0.3889 -0.0000  0.4377  0.9948  1.4293
#>  0.0000 -0.3800 -0.9545  0.0000  0.0332
#> -0.0000 -0.0000  0.0000 -0.0000 -0.0000
#> -0.0000  0.0000 -0.0651  0.0000  0.0000
#>  2.1180 -0.0000  0.0000  0.0000  0.0000
#> -0.0000 -0.0000 -1.2180 -0.3652  0.1876
#>  0.0000 -0.0014 -0.1591  0.8970  1.4790
#>  0.0000 -0.0000 -0.0000  0.7075 -0.0244
#> -0.0114  0.0000 -0.1660 -0.0000 -0.8783
#> -0.0000  0.0000  0.4622 -1.2290 -0.0000
#> [ CPUFloatType{10,5} ]
```
