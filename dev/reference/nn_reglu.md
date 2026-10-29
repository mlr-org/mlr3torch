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
#> -0.2808  0.0000 -0.2490  0.8503  0.0000
#> -0.0491  1.2793 -0.0000  0.0000 -0.0000
#> -0.0000 -0.0130 -0.0144  0.9730 -0.2072
#> -0.0000 -1.2450  0.0993  0.0000  0.0000
#>  3.1025  0.1854  0.0000  0.7055 -0.0000
#>  0.0000  1.9519 -0.0420 -0.4866 -0.2745
#> -0.0000 -0.0162 -0.0000 -0.3598  0.0043
#> -0.0000 -0.6953 -0.1855 -0.0000 -0.0000
#> -0.0000  3.3057 -0.0937  1.4415  0.0000
#> -0.0000  0.0000 -0.2820 -1.5265 -0.0000
#> [ CPUFloatType{10,5} ]
```
