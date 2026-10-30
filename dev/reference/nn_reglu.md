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
#> -0.0000 -0.0490 -0.0000 -0.0065  0.0000
#>  0.0000 -0.0000  0.0000  0.0000 -0.3440
#>  0.0000 -0.0000 -0.2226  0.0000  0.0000
#>  0.0000  0.1998  0.0000 -0.1089  0.0000
#>  0.0000 -1.5342 -0.0000  0.3410 -0.0000
#>  0.0000  0.0000  1.3321 -0.0000  0.0000
#>  0.0000 -0.0000  0.0000 -0.0000  0.0000
#> -0.0000 -0.0000  0.1341 -0.1954 -0.0000
#>  0.0000 -0.0281  0.0000 -0.0000  0.5763
#> -0.0164  2.0203 -0.0000 -0.0000  0.0000
#> [ CPUFloatType{10,5} ]
```
