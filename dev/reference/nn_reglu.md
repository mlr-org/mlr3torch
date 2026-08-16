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
#>  0.0000 -1.3267 -0.0025  0.0000  0.0000
#> -0.0000  0.0000 -1.3893  0.0000 -0.8705
#>  0.0000  0.8514  0.4186 -0.0000 -0.9933
#> -0.0000  0.1016 -0.0000 -0.0000 -0.0000
#>  0.0000  0.0000  0.0077  0.7245  0.0000
#>  0.0000  0.3453  0.0000 -0.0000  0.0000
#>  0.0000 -0.0000  0.0000 -0.0000  0.0000
#>  0.4864  0.1762 -0.0000 -0.0000  0.0526
#> -0.5304  0.0000  2.3163  0.3972  0.0000
#>  0.3531 -0.0000  3.2053  0.0000 -0.0694
#> [ CPUFloatType{10,5} ]
```
