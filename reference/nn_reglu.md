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
#> -0.1648 -0.0000  0.8742 -0.2492 -0.0000
#>  0.0000 -0.7003 -0.0000 -0.7794  0.0434
#> -0.0000 -0.0000  0.0000 -0.1540  0.0000
#>  0.1109 -0.4921 -0.1296  0.0000  0.9377
#> -0.0000 -0.5097  0.7377  0.6484 -0.4211
#> -0.0000 -0.0235  0.0000 -0.0000 -0.3357
#> -0.0000 -0.0030 -0.0000 -0.0568 -0.1143
#>  0.0192  0.0000 -0.2485 -3.0636 -2.0717
#> -0.0000  1.1869 -0.0179 -0.0686  0.0000
#>  0.2612 -2.4636 -1.1190 -0.0000  0.1535
#> [ CPUFloatType{10,5} ]
```
