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
#> -0.5882  0.0000  0.6729  0.0000 -0.0000
#>  0.0000  0.2348 -0.0000  1.4237  0.0000
#> -0.0000  0.0000  0.0000  0.0000 -0.1129
#> -0.9024 -0.0000 -0.3051  0.0000  0.0000
#> -0.0000 -0.0000 -0.0000 -0.0000 -0.0000
#> -0.4470 -0.0000  0.0028  1.6916 -0.0000
#> -0.0000  0.3575  0.0000  0.5203  0.0000
#>  0.0000 -0.2454 -0.0000  0.1075  0.3726
#>  0.0000 -0.0000 -0.3596  0.2835 -0.0000
#> -0.0000 -0.0310  0.0000 -0.2920  2.7228
#> [ CPUFloatType{10,5} ]
```
