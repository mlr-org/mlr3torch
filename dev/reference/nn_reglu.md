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
#>  0.2486  0.3678 -0.1035  1.2982  0.0000
#>  0.0000  0.7163 -0.0000  0.0159  0.0000
#> -0.5025 -0.2145 -0.0000 -4.6203 -0.0000
#>  0.0000  0.0000 -0.0000  0.4465 -0.0000
#>  0.3149  0.8179  0.0000 -0.0675  0.3787
#> -0.0000 -1.0827  0.0982 -0.0000  0.8184
#>  0.0000 -1.2131 -0.0000 -0.2018 -0.7913
#> -0.0000  0.5389  0.0000 -0.1882 -1.7094
#> -1.8931 -0.0000 -0.0000  0.0000  0.0000
#> -3.2154  0.0000 -0.0000 -0.0000 -0.0000
#> [ CPUFloatType{10,5} ]
```
