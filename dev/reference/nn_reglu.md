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
#> -0.0000 -0.0000  0.0001 -0.0000  0.6465
#> -0.0000 -0.7402 -0.0000  0.0000 -0.0000
#> -0.0000  0.0000 -0.0000  0.6876  0.0000
#>  0.0000  3.9683  0.0000  0.0000  0.0826
#> -0.0000 -0.0000  0.0000  0.7516  1.5979
#>  0.0000  0.0000 -0.0000 -0.4540 -0.0000
#>  0.0000  0.0000  0.1352 -0.1596 -0.0000
#> -5.0548 -0.0000 -0.0918  0.0000  0.0000
#> -0.6009  0.0000 -0.0000 -0.2062  0.0000
#>  0.1914 -0.4039  0.0000 -0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
