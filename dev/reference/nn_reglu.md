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
#>  0.0000 -0.0000  0.0000 -1.2507  0.0000
#> -0.3443  0.0000  0.0000 -0.7691  0.0000
#> -0.4355  0.0000  0.7381 -0.0000  0.0448
#>  1.2989  0.0000 -0.0000 -0.4854 -0.0000
#>  0.0000  0.0000 -0.5286  0.0000  0.7381
#>  0.0000  0.7640 -0.0000 -0.0000  0.5770
#> -0.4899  0.8355  0.6363 -0.0346  0.0166
#> -0.0000 -0.8613 -2.0120 -0.0000  0.0000
#>  0.0000 -0.0000  0.1383 -0.3570 -0.0000
#> -0.0031 -0.0000 -0.6349  0.0000 -0.1829
#> [ CPUFloatType{10,5} ]
```
