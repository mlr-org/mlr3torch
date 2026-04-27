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
#> -0.9222 -0.0000 -1.7133  0.0000  0.5661
#>  0.0000  0.0000  0.0299  0.1066  0.0827
#>  2.2558  0.9657  1.5661 -0.1450 -0.3197
#>  0.0000  0.7779 -0.0000  0.0909  1.5486
#>  0.0000  0.0942  0.0000  0.0000  0.4536
#>  1.4871  0.0930  0.1365  0.2125 -0.1506
#> -0.1897 -0.0362 -0.0000  0.0000 -0.2446
#> -0.0000  0.3593 -0.0000  1.4197 -0.0000
#>  3.5929 -0.0000 -0.0000  0.5571 -0.0000
#> -0.0321 -0.0000 -0.0000  0.4033 -0.0000
#> [ CPUFloatType{10,5} ]
```
