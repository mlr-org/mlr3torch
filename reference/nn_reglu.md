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
#> -0.0000  0.1494  0.3815  0.0000  0.0000
#>  0.0000 -0.6794 -0.0000 -0.0000 -0.0000
#> -0.0000  1.3765  0.0285 -0.0000  0.0774
#> -1.0112  0.0838  0.2787  0.0000  0.0463
#>  0.0000  0.0000 -0.1545  0.0703 -0.0000
#> -0.0000  0.0000 -0.0000 -1.0270  0.0000
#>  0.5356 -0.2622  0.0000 -0.3265 -0.0000
#> -0.2804  1.8972 -0.6192  0.0000  0.0000
#> -0.0000 -0.3528  0.2457  0.2203  0.4357
#>  0.3625  0.0000  0.5170 -0.8212 -0.0015
#> [ CPUFloatType{10,5} ]
```
