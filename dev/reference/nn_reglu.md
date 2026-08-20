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
#>  0.0000  0.0000  0.1596  0.0469 -0.0000
#> -0.0621 -0.0619 -0.0000 -0.0514 -0.0000
#> -0.0000  0.0445 -0.0000 -0.1375 -0.0395
#>  0.1247  0.0000  0.0000 -0.1292  0.0000
#> -0.0379  0.0000 -0.2175  0.0000  0.5071
#> -0.4504  1.7836 -0.0000 -0.4446 -0.0000
#> -0.0024  0.2196  0.0000 -0.0000  0.3082
#>  0.0945 -0.0000  0.0000 -0.0577 -0.0000
#> -0.0000  0.0000  0.4882 -0.0000  0.0000
#>  0.0000  0.0000 -0.3572 -1.5469  0.8401
#> [ CPUFloatType{10,5} ]
```
