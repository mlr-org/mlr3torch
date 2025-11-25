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
#> -0.0000 -0.0197 -0.0000  0.2292  0.0000
#>  0.1167  0.0125  0.0000  1.3586  0.0000
#>  0.3018  1.5760  0.0000 -0.0317  0.0000
#>  0.0062  0.3015 -0.0000  0.0000  0.7823
#>  0.0000 -0.0000 -0.3295  0.0000  0.0000
#>  0.0000 -0.0000  0.0000  1.1068 -0.0000
#> -0.2054  0.2330 -0.0000 -0.0000 -0.0000
#>  0.0270  0.4100  1.1754 -1.5307  0.0000
#> -0.2954  0.0000 -0.0000 -1.0582  0.5815
#> -0.0276  0.0000  0.2980  0.3403  0.0427
#> [ CPUFloatType{10,5} ]
```
