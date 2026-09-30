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
#>  0.0000  0.3565 -0.6179  0.0000 -0.7844
#> -0.4128 -0.0210 -0.6816  0.0000 -0.5630
#> -0.0000 -0.5357 -1.3473 -0.3497  0.0000
#> -0.0000 -0.0506 -0.0185  0.0000 -0.0081
#>  0.0000  0.0000 -0.0000  0.0000 -0.0000
#> -0.0000  0.0000  0.0000  1.7590  0.0000
#>  0.3801  0.5128 -0.0020 -1.4613  0.0000
#> -2.4758 -0.0000  0.0000 -0.0000 -0.0000
#> -0.0000 -0.2863 -0.0003 -0.0000  0.0000
#>  0.2271 -0.4249 -0.0000 -0.3257 -0.2456
#> [ CPUFloatType{10,5} ]
```
