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
#> -0.0000 -1.9962  0.0601  0.0000  0.0000
#>  0.2062 -0.4612  0.0907  0.0000 -1.8944
#>  0.0000 -0.0000 -0.0000  0.5022  0.5512
#>  1.0672 -0.3578  0.0000 -0.0969  0.0000
#>  0.0873  0.0000 -1.2107  1.0908 -0.0000
#> -0.0000  0.0000 -0.0000  0.0000 -0.2490
#> -0.4551 -0.0000  1.3199 -0.0000 -0.0000
#>  0.0000 -0.0000 -1.0258 -0.0000 -0.0000
#>  1.1784  3.5426 -1.0370 -0.0482  0.9200
#>  1.3295  0.0000  0.0000 -0.0000  0.0093
#> [ CPUFloatType{10,5} ]
```
